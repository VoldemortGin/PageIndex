from __future__ import annotations

import asyncio
import importlib.util
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import Mock

import pytest

from conftest import build_pdf

MODULE_PATH = Path(__file__).resolve().parents[1] / 'notebooks' / 'batch_qa_support.py'
spec = importlib.util.spec_from_file_location('batch_qa_support', MODULE_PATH)
support = importlib.util.module_from_spec(spec)
spec.loader.exec_module(support)


def test_nested_fields_and_scope_do_not_leak_answers(tmp_path):
    path = tmp_path / 'golden.json'
    path.write_text(json.dumps({'records': [{'uid': 'x', 'prompt': 'Revenue?', 'answers': [{'text': '1234'}], 'files': [{'name': 'Report.pdf'}]}]}))
    question = support.load_questions(path, {'id': 'uid', 'question': 'prompt', 'expected': 'answers[].text', 'doc': 'files[].name'}, 'records')[0]
    assert question['expected'] == '1234'
    assert question['question'] == 'Revenue?'
    documents = [{'doc_id': 'a', 'source': 'Reports/Report.pdf', 'name': 'Report.pdf'}]
    assert support.resolve_scope(question['doc'], documents) == ['a']
    with pytest.raises(ValueError, match='matched 0'):
        support.resolve_scope(['missing.pdf'], documents)
    with pytest.raises(ValueError, match='matched 2'):
        support.resolve_scope(['Report.pdf'], documents + [{**documents[0], 'doc_id': 'b'}])


def test_ingest_resume_requires_same_content_and_settings(tmp_path):
    pdf_dir = tmp_path / 'pdf'
    pdf_dir.mkdir()
    pdf = pdf_dir / 'report.pdf'
    pdf.write_bytes(build_pdf(['one']))
    client = Mock()
    client.submit_document.side_effect = [{'doc_id': 'one'}, {'doc_id': 'two'}, {'doc_id': 'three'}]
    client.is_retrieval_ready.return_value = True
    args = (client, pdf_dir, tmp_path / 'work', 'flash')
    assert support.ingest(*args, {'model': 'a'})[0]['status'] == 'indexed'
    assert support.ingest(*args, {'model': 'a'})[0]['status'] == 'reused'
    pdf.write_bytes(build_pdf(['two']))
    assert support.ingest(*args, {'model': 'a'})[0]['doc_id'] == 'two'
    assert support.ingest(*args, {'model': 'b'})[0]['doc_id'] == 'three'
    assert client.submit_document.call_count == 3


def test_batch_resume_recomputes_changed_question(monkeypatch, tmp_path):
    question = {'id': 'x', 'question': 'One?', 'expected': 'gold', 'doc': []}
    documents = [{'doc_id': 'a', 'source': 'a.pdf', 'name': 'a.pdf', 'status': 'indexed'}]
    calls = []
    def answer(client, q, docs, signature=None, **kwargs):
        calls.append(q['question'])
        return {**q, 'error': None, 'seconds': 0, 'status': 'success', 'fingerprint': support.question_fingerprint(q, signature, docs)}
    monkeypatch.setattr(support, 'answer_question', answer)
    args = (None, [question], documents, tmp_path / 'results.jsonl', 'config')
    support.run_batch(*args)
    documents[0]['status'] = 'reused'
    support.run_batch(*args, resume=True)
    question['question'] = 'Two?'
    support.run_batch(*args, resume=True)
    assert calls == ['One?', 'Two?']


def test_native_deadline_cancels_and_closes_generator(monkeypatch):
    from pageindex import local_chat
    closed = []
    monkeypatch.setattr(local_chat, '_chat_agent', lambda *a, **k: (None, None, None))
    async def events(*args):
        try:
            yield {'type': 'answer', 'delta': 'partial'}
            await asyncio.sleep(10)
        finally:
            closed.append(True)
    monkeypatch.setattr(local_chat, '_chat_events_agen', events)
    received, error = support.native_events(None, 'question', [], 0.01, 25, None, {})
    assert received == [{'type': 'answer', 'delta': 'partial'}]
    assert 'TimeoutError' in error
    assert closed == [True]


def test_real_native_agent_tools_and_running_loop(tmp_path, monkeypatch):
    monkeypatch.setenv('LITELLM_LOCAL_MODEL_COST_MAP', 'True')
    monkeypatch.setenv('OPENAI_AGENTS_DISABLE_TRACING', '1')
    from pageindex import PageIndexClient
    from pageindex.local_store import DocStore
    requests = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            requests.append(body)
            if len(requests) == 1:
                delta = {'role': 'assistant', 'tool_calls': [{'index': 0, 'id': 'call_1', 'type': 'function', 'function': {'name': 'get_page_content', 'arguments': json.dumps({'doc_name': 'report.pdf', 'pages': '1', 'folder_id': None, 'wait_for_completion': False})}}]}
                finish = 'tool_calls'
            else:
                delta = {'role': 'assistant', 'content': 'Revenue was 1234. <cite doc="report.pdf" page="1"/>'}
                finish = 'stop'
            base = {'id': 'chatcmpl-test', 'object': 'chat.completion.chunk', 'created': 0, 'model': body['model']}
            chunks = [{**base, 'choices': [{'index': 0, 'delta': delta, 'finish_reason': None}]}, {**base, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': finish}]}]
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.end_headers()
            for chunk in chunks:
                self.wfile.write(f'data: {json.dumps(chunk)}\n\n'.encode())
            self.wfile.write(b'data: [DONE]\n\n')
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    store_path = tmp_path / 'store'
    store = DocStore(str(store_path))
    store.save_document('pi-test', {'id': 'pi-test', 'name': 'report.pdf', 'description': 'Revenue report', 'status': 'completed', 'createdAt': '2026-01-01', 'pageNum': 1, 'folderId': None, 'metadata': None, 'mode': 'flash'}, [{'node_id': '0000', 'title': 'Revenue', 'start_index': 1, 'end_index': 1}], [{'page_index': 1, 'markdown': 'Revenue was 1234.'}])
    client = PageIndexClient(mode='local', index_model='openai/test', chat_model='openai/test', storage_path=str(store_path), chat_backend={'api_key': 'fake', 'base_url': f'http://127.0.0.1:{server.server_port}/v1'})
    question = {'id': 'x', 'question': 'Revenue?', 'expected': 'SECRET_GOLDEN_1234', 'doc': ['report.pdf']}
    documents = [{'doc_id': 'pi-test', 'source': 'report.pdf', 'name': 'report.pdf'}]
    async def run():
        return support.answer_question(client, question, documents, timeout=15)
    try:
        row = asyncio.run(run())
    finally:
        server.shutdown()
        server.server_close()
    assert row['status'] == 'success', row['error']
    assert '1234' in row['answer']
    assert row['pages_read'] == ['report.pdf:1']
    assert 'Revenue was 1234' in json.dumps(row['retrieved_content'])
    assert row['citations'][0]['page'] == 1
    assert 'SECRET_GOLDEN' not in json.dumps(requests)
    assert len(requests) == 2


def test_original_source_alias_preserved_after_native_name_uniquing():
    documents = [
        {'doc_id': 'a', 'source': 'Reports/report.pdf', 'name': 'report_1.pdf'},
        {'doc_id': 'b', 'source': 'Reports/other.pdf', 'name': 'other.pdf'},
    ]
    assert support.resolve_scope(['report.pdf'], documents) == ['a']
    assert support.resolve_scope(['Reports/report.pdf'], documents) == ['a']
    assert support.resolve_scope(['report'], documents) == ['a']
    with pytest.raises(ValueError, match='matched 2'):
        support.resolve_scope(['report.pdf'], documents + [
            {'doc_id': 'c', 'source': 'Other/report.pdf', 'name': 'report_2.pdf'}
        ])


def test_failed_page_tool_is_attempted_but_never_recorded_as_read(monkeypatch):
    events = [
        {'type': 'tool_call', 'name': 'get_page_content', 'arguments': {'doc_name': 'report.pdf', 'pages': '99'}, 'call_id': 'x'},
        {'type': 'tool_result', 'name': 'get_page_content', 'output': json.dumps({'success': False, 'error': 'Invalid page'}), 'call_id': 'x'},
        {'type': 'answer', 'delta': 'Page unavailable'},
    ]
    monkeypatch.setattr(support, 'native_events', lambda *args: (events, None))
    client = Mock()
    client.get_citations.return_value = []
    question = {'id': 'x', 'question': 'Read 99', 'doc': ['report.pdf']}
    documents = [{'doc_id': 'a', 'source': 'report.pdf', 'name': 'report.pdf'}]
    row = support.answer_question(client, question, documents)
    assert row['pages_attempted'] == ['report.pdf:99']
    assert row['pages_read'] == []
    assert len(row['tool_results']) == 1


def encrypted_pdf(path, password):
    from io import BytesIO
    from PyPDF2 import PdfReader, PdfWriter

    writer = PdfWriter()
    writer.append(PdfReader(BytesIO(build_pdf(['Encrypted content']))))
    writer.add_outline_item('Chapter', 0)
    writer.encrypt(password, owner_password='owner-secret')
    with path.open('wb') as stream:
        writer.write(stream)


@pytest.mark.parametrize('password', ['', ' spaced secret '])
def test_ingest_decrypts_without_changing_source_and_preserves_outline(tmp_path, password):
    from PyPDF2 import PdfReader
    import pypdfium2

    pdf_dir = tmp_path / 'pdf'
    pdf_dir.mkdir()
    original = pdf_dir / 'report.pdf'
    encrypted_pdf(original, password)
    original_bytes = original.read_bytes()
    submitted_paths = []

    def submit(path, mode):
        submitted = Path(path)
        submitted_paths.append(submitted)
        assert submitted.name == original.name
        reader = PdfReader(path)
        assert not reader.is_encrypted
        assert 'Encrypted content' in reader.pages[0].extract_text()
        assert reader.outline[0]['/Title'] == 'Chapter'
        with pypdfium2.PdfDocument(path) as document:
            assert len(document) == 1
        return {'doc_id': 'one', 'name': original.name}

    client = Mock()
    client.submit_document.side_effect = submit
    client.is_retrieval_ready.return_value = True
    args = (client, pdf_dir, tmp_path / 'work', 'flash', {})
    assert support.ingest(*args, pdf_passwords=['bad', password])[0]['status'] == 'indexed'
    assert original.read_bytes() == original_bytes
    assert not submitted_paths[0].exists()
    assert not submitted_paths[0].parent.exists()
    assert support.ingest(*args, pdf_passwords=[])[0]['status'] == 'reused'
    assert client.submit_document.call_count == 1


def test_wrong_password_is_retried_and_batch_continues(tmp_path, capsys):
    pdf_dir = tmp_path / 'pdf'
    pdf_dir.mkdir()
    encrypted_pdf(pdf_dir / 'a.pdf', ' secret ')
    (pdf_dir / 'b.pdf').write_bytes(build_pdf(['Plain text']))
    client = Mock()
    client.submit_document.return_value = {'doc_id': 'plain'}
    client.is_retrieval_ready.return_value = True
    args = (client, pdf_dir, tmp_path / 'work', 'flash', {})
    assert len(support.ingest(*args, pdf_passwords=['wrong-secret'])) == 1
    report = json.loads((tmp_path / 'work' / 'ingest_report.json').read_text())
    assert report[0]['status'] == 'skipped_wrong_password'
    assert 'wrong-secret' not in json.dumps(report) + capsys.readouterr().out
    client.submit_document.return_value = {'doc_id': 'encrypted'}
    assert len(support.ingest(*args, pdf_passwords=[' secret '])) == 2
    assert client.submit_document.call_count == 2


def test_decrypted_temporary_file_cleaned_after_submit_error(tmp_path):
    pdf_dir = tmp_path / 'pdf'
    pdf_dir.mkdir()
    encrypted_pdf(pdf_dir / 'report.pdf', 'secret')
    submitted_paths = []

    def submit(path, mode):
        submitted_paths.append(Path(path))
        raise RuntimeError('Indexing failed')

    client = Mock()
    client.submit_document.side_effect = submit
    assert support.ingest(client, pdf_dir, tmp_path / 'work', 'flash', {}, pdf_passwords=['secret']) == []
    assert not submitted_paths[0].parent.exists()


def test_plain_pdf_is_submitted_directly(tmp_path):
    path = tmp_path / 'plain.pdf'
    path.write_bytes(build_pdf(['Plain content']))
    with support.prepared_pdf(path, []) as prepared:
        assert prepared == path
    assert path.exists()


def test_password_candidates_preserve_whitespace_and_primary_priority():
    assert support.pdf_password_candidates(' primary ', '["second", " primary ", ""]') == [' primary ', 'second']
    assert support.pdf_password_candidates(None, ' whole secret ') == [' whole secret ']
    with pytest.raises(ValueError, match='only strings'):
        support.pdf_password_candidates(None, '[123]')


def test_aes_encrypted_pdf_can_be_prepared(tmp_path):
    pypdf = pytest.importorskip('pypdf')
    pytest.importorskip('Crypto')
    from io import BytesIO
    from PyPDF2 import PdfReader

    writer = pypdf.PdfWriter()
    writer.append(pypdf.PdfReader(BytesIO(build_pdf(['AES content']))))
    writer.encrypt(' aes secret ', algorithm='AES-256')
    path = tmp_path / 'aes.pdf'
    with path.open('wb') as stream:
        writer.write(stream)
    with support.prepared_pdf(path, ['bad', ' aes secret ']) as prepared:
        reader = PdfReader(prepared)
        assert not reader.is_encrypted
        assert 'AES content' in reader.pages[0].extract_text()
    assert not prepared.parent.exists()
