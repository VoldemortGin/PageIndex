from __future__ import annotations

import asyncio
import hashlib
import json
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Any

DEFAULT_FIELD_MAP = {'id': 'id', 'question': 'question', 'expected': 'expected', 'doc': 'doc', 'pages': None}


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def field_value(item: Any, path: str | None) -> Any:
    if not path:
        return None
    if isinstance(item, dict) and path in item:
        return item[path]

    def walk(node, segments):
        if not segments:
            return node
        token, rest = segments[0], segments[1:]
        spread = token.endswith('[]')
        key = token[:-2] if spread else token
        if key:
            if not isinstance(node, dict) or key not in node:
                return None
            node = node[key]
        if not spread:
            return walk(node, rest)
        if not isinstance(node, list):
            return None
        result = []
        for element in node:
            value = walk(element, rest)
            result.extend(value if isinstance(value, list) else [value])
        return result

    value = walk(item, path.split('.'))
    if isinstance(value, list) and '[]' in path:
        unique = {}
        for entry in value:
            if entry is not None and entry != '':
                unique.setdefault(json.dumps(entry, sort_keys=True), entry)
        return list(unique.values())
    return value


def load_questions(path: Path, field_map: dict[str, Any] | None = None, records_key: str | None = None) -> list[dict[str, Any]]:
    mapping = DEFAULT_FIELD_MAP if field_map is None else field_map
    text = path.read_text(encoding='utf-8-sig')
    data = [json.loads(line) for line in text.splitlines() if line.strip()] if path.suffix.lower() == '.jsonl' else json.loads(text)
    if records_key:
        data = field_value(data, records_key)
    elif isinstance(data, dict):
        data = data.get('questions')
    if not isinstance(data, list):
        raise ValueError('Dataset must contain a list; configure RECORDS_KEY for nested records.')
    questions, seen = [], set()
    for index, item in enumerate(data, 1):
        if isinstance(item, str):
            item = {mapping.get('question') or 'question': item}
        if not isinstance(item, dict):
            raise ValueError(f'Question {index} must be an object or string')
        row = {key: field_value(item, mapping.get(key)) for key in DEFAULT_FIELD_MAP}
        for key in ('id', 'question', 'expected'):
            if isinstance(row[key], list):
                row[key] = '\n'.join(str(v) for v in row[key])
        row['id'] = str(row['id'] or f'Q{index:03d}')
        row['question'] = str(row['question'] or '').strip()
        if not row['question']:
            continue
        if row['id'] in seen:
            raise ValueError(f"Duplicate question id: {row['id']}")
        seen.add(row['id'])
        docs = row['doc'] if isinstance(row['doc'], list) else re.split(r'[;|]', str(row['doc'] or ''))
        row['doc'] = [str(v).strip() for v in docs if str(v).strip().casefold() not in ('', '*', 'all', 'all docs', 'all_docs')]
        row['extra'] = {key: value for key, value in item.items() if key not in {v for v in mapping.values() if v and not re.search(r'[.\[\]]', v)}}
        questions.append(row)
    return questions


def resolve_scope(hints: list[str], documents: list[dict[str, Any]]) -> list[str]:
    if not hints:
        return [d['doc_id'] for d in documents]
    out = []
    for hint in hints:
        key = hint.replace('\\', '/').casefold()
        exact = [d for d in documents if key in {str(d.get(k, '')).replace('\\', '/').casefold() for k in ('source', 'name', 'doc_id')} or key in {Path(d['name']).stem.casefold(), Path(d['source']).stem.casefold(), Path(d['source']).name.casefold()}]
        matches = exact or [d for d in documents if key in d['source'].casefold() or key in d['name'].casefold()]
        if len(matches) != 1:
            raise ValueError(f'Document hint {hint!r} matched {len(matches)} documents; refusing ambiguous/full-library fallback')
        if matches[0]['doc_id'] not in out:
            out.append(matches[0]['doc_id'])
    return out


def read_results(path: Path) -> dict[str, dict[str, Any]]:
    latest = {}
    if path.is_file():
        for line in path.read_text(encoding='utf-8').splitlines():
            try:
                row = json.loads(line)
                latest[str(row['id'])] = row
            except (ValueError, KeyError):
                continue
    return latest


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    temporary.replace(path)


def retry(operation, retries: int, wait: float):
    for attempt in range(retries + 1):
        try:
            return operation()
        except Exception as exc:
            if attempt == retries or not any(token in str(exc).lower() for token in ('429', 'rate limit', 'rate_limit', 'ratelimit')):
                raise
            time.sleep(wait)


def ingest(client, pdf_dir: Path, work_dir: Path, mode: str, index_signature: Any, force: bool = False,
           retries: int = 5, wait: float = 60.0) -> list[dict[str, Any]]:
    if not pdf_dir.is_dir():
        raise FileNotFoundError(f'PDF_DIR does not exist: {pdf_dir}')
    manifest_path = work_dir / 'ingest_manifest.json'
    old = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    rows, current = [], {}
    paths = sorted(p for p in pdf_dir.rglob('*') if p.is_file() and p.suffix.lower() == '.pdf')
    if not paths:
        raise ValueError(f'No PDFs under {pdf_dir}')
    for path in paths:
        source = path.relative_to(pdf_dir).as_posix()
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        signature = fingerprint([source, digest.hexdigest(), mode, index_signature])
        row = {'source': source, 'name': path.name, 'signature': signature}
        prior = old.get(source, {})
        try:
            reusable = not force and prior.get('signature') == signature and prior.get('doc_id')
            if reusable:
                try:
                    reusable = client.is_retrieval_ready(prior['doc_id'])
                except Exception:
                    reusable = False
            if reusable:
                row.update(doc_id=prior['doc_id'], name=prior.get('name', path.name), status='reused')
            else:
                submitted = retry(lambda: client.submit_document(str(path), mode=mode), retries, wait)
                row.update(doc_id=submitted['doc_id'], name=submitted.get('name', path.name), status='indexed')
            current[source] = row
        except Exception as exc:
            row.update(status='error', error=str(exc))
        rows.append(row)
        write_json(manifest_path, current)
        write_json(work_dir / 'ingest_report.json', rows)
        print(f"{source}: {row['status']}")
    return [row for row in rows if row.get('doc_id')]


def native_events(client, question: str, scope: list[str], timeout: float, max_turns: int | None,
                  reasoning_effort: str | None, extra_headers: dict[str, str]):
    # Centralized adapter to this checkout's native chat primitive enables a total
    # deadline with cancellation, including stalled backend requests, on Python 3.10.
    from pageindex.local_chat import _chat_agent, _chat_events_agen, _run_kwargs
    from pageindex.utils import run_off_loop

    events = []

    async def run():
        agent, items, _ = _chat_agent(client, [{'role': 'user', 'content': question}], scope, None,
                                      reasoning_effort=reasoning_effort, extra_headers=extra_headers)
        generator = _chat_events_agen(client, agent, items, _run_kwargs(max_turns))

        async def consume():
            try:
                async for event in generator:
                    events.append(event)
            finally:
                await generator.aclose()

        await asyncio.wait_for(consume(), timeout=timeout)

    try:
        run_off_loop(asyncio.run, run())
        return events, None
    except Exception as exc:
        return events, f'{type(exc).__name__}: {exc}'


def question_fingerprint(question, signature, documents):
    stable_docs = [{key: document.get(key) for key in ('doc_id', 'source', 'signature')} for document in documents]
    return fingerprint([question, signature, stable_docs])


def page_result(output):
    if isinstance(output, list):
        for item in output:
            result = page_result(item)
            if result:
                return result
        return None
    if isinstance(output, str):
        try:
            return page_result(json.loads(output))
        except ValueError:
            return None
    if isinstance(output, dict):
        if output.get('type') == 'text':
            return page_result(output.get('text', ''))
        return output if output.get('success') else None
    return None


def answer_question(client, question: dict[str, Any], documents: list[dict[str, Any]], timeout: float = 600,
                    max_turns: int | None = 25, reasoning_effort: str | None = None,
                    extra_headers: dict[str, str] | None = None, signature: Any = None) -> dict[str, Any]:
    started = time.monotonic()
    row = {**question, 'answer': '', 'error': None, 'tool_calls': [], 'tool_results': [], 'pages_read': [], 'pages_attempted': [], 'retrieved_content': [], 'citations': []}
    row['fingerprint'] = question_fingerprint(question, signature, documents)
    try:
        scope = resolve_scope(question['doc'], documents)
        if not scope:
            raise ValueError('No successfully ingested documents available')
        row['scope'] = scope
        events, error = native_events(client, question['question'], scope, timeout, max_turns, reasoning_effort, extra_headers or {})
        row['error'] = error
        for event in events:
            kind = event['type']
            if kind == 'answer':
                row['answer'] += event['delta']
            elif kind == 'tool_call':
                row['tool_calls'].append(event)
                if event['name'] == 'get_page_content':
                    args = event.get('arguments', {})
                    if isinstance(args, dict):
                        row['pages_attempted'].append(f"{args.get('doc_name', args.get('doc_id', '?'))}:{args.get('pages', '?')}")
            elif kind == 'tool_result':
                row['tool_results'].append(event)
                if event['name'] == 'get_page_content':
                    row['retrieved_content'].append(event['output'])
                    payload = page_result(event['output'])
                    if payload:
                        for page in payload.get('content', []):
                            if isinstance(page, dict) and page.get('page') is not None:
                                value = f"{payload.get('doc_name', '?')}:{page['page']}"
                                if value not in row['pages_read']:
                                    row['pages_read'].append(value)
        if not row['error'] and not row['answer'].strip():
            row['error'] = 'Model returned no answer'
        if row['answer']:
            row['citations'] = client.get_citations(row['answer'])
    except Exception as exc:
        row['error'] = f'{type(exc).__name__}: {exc}'
    row.update(seconds=round(time.monotonic() - started, 3), status='error' if row['error'] else 'success')
    return row


def run_batch(client, questions: list[dict[str, Any]], documents: list[dict[str, Any]], results_path: Path,
              signature: Any, concurrency: int = 1, resume: bool = False, retries: int = 5,
              wait: float = 60, checkpoint=None, **kwargs) -> list[dict[str, Any]]:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    latest = read_results(results_path) if resume else {}
    if not resume:
        results_path.write_text('', encoding='utf-8')
    lock = Lock()

    def one(question):
        key = question['id']
        expected_fingerprint = question_fingerprint(question, signature, documents)
        if latest.get(key, {}).get('status') == 'success' and latest[key].get('fingerprint') == expected_fingerprint:
            return
        for attempt in range(retries + 1):
            row = answer_question(client, question, documents, signature=signature, **kwargs)
            if not row['error'] or not any(t in row['error'].lower() for t in ('429', 'rate limit', 'rate_limit', 'ratelimit')) or attempt == retries:
                break
            time.sleep(wait)
        row['rate_limit_retries'] = attempt
        with lock:
            latest[key] = row
            with results_path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(row, ensure_ascii=False, default=str) + '\n')
                stream.flush()
            if checkpoint is not None:
                checkpoint()
            print(f"{key}: {row['status']} ({row['seconds']}s)")

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        list(pool.map(one, questions))
    return [latest[q['id']] for q in questions if q['id'] in latest]


def sync_directory(source: Path, destination: Path | None) -> None:
    if destination is not None and source.resolve() != destination.resolve():
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, destination, dirs_exist_ok=True)
