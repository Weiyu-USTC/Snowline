"""Local-only dashboard server for the existing snowfall research models.

Refresh means fetching new forecast inputs and running saved weights. It does
not refit a model, change the frozen benchmark, or schedule background updates.
"""
from __future__ import annotations

import argparse
import errno
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import threading
import time
from urllib.parse import parse_qs, urlparse
from urllib.request import urlopen
import uuid
import webbrowser

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / 'dashboard'
SITES = {'alta', 'vail', 'mansfield', 'jay'}
JOBS = {}
LOCK = threading.RLock()
WORKER = ThreadPoolExecutor(max_workers=1, thread_name_prefix='forecast-refresh')


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def attach_radar(payload):
    if payload.get('site', {}).get('id') == 'alta':
        try:
            from dashboard_alta import get_alta_nowcast
            payload['radar'] = get_alta_nowcast()
        except Exception as exc:
            payload['radar'] = {'status': 'unavailable', 'warnings': [str(exc)], 'models': []}
    return payload


def attach_neural48(payload):
    from dashboard_neural48 import read_neural48
    payload['neural48'] = read_neural48(payload.get('site', {}).get('id'))
    if payload.get('status') == 'unavailable' and payload['neural48'].get('status') != 'unavailable':
        payload['status'] = 'partial'
    return payload


def read_forecast(site):
    from dashboard_forecasts import get_forecast
    payload=attach_neural48(attach_radar(get_forecast(site, refresh=False)))
    try:
        from dashboard_portfolio import read_deployed
        payload['deployed']=read_deployed(site)
        if payload.get('status')=='unavailable' and payload['deployed'].get('status')!='unavailable':
            payload['status']='partial'
    except Exception as exc:
        payload['deployed']={'status':'unavailable','site_id':site,'horizons':[],
                             'warnings':[f'The selected model forecast could not be read: {exc}']}
    return payload


def set_job(job_id, **changes):
    with LOCK:
        JOBS[job_id].update(changes, updated_at=utc_now())


def run_refresh_all(job_id):
    try:
        set_job(job_id,status='running',message='Updating radar, HRRR and the models for all four mountains…')
        from refresh_deployed_forecasts import refresh_all
        def progress(message,source=None,source_status=None):
            with LOCK:
                job=JOBS[job_id];job.update(message=str(message),updated_at=utc_now())
                if source:job.setdefault('sources',{})[source]={'status':source_status,'message':str(message)}
        outcome=refresh_all(progress=progress)
        results={};errors=list(outcome.get('errors',[]))
        for site in ['alta','vail','mansfield','jay']:
            try:results[site]=read_forecast(site)
            except Exception as exc:
                results[site]={'site':{'id':site},'status':'unavailable','errors':[str(exc)]}
                errors.append({'source':site,'message':str(exc)})
        status=outcome['status']
        if errors and status=='succeeded':status='partial'
        if all(r.get('status')=='unavailable' for r in results.values()):status='failed'
        message='All four mountains updated.' if status=='succeeded' else outcome.get('message','Some sources could not update. Available forecasts and their original times are shown.')
        result={'sites':results,'sources':outcome.get('sources',{}),'errors':errors,'status':status}
        set_job(job_id,status=status,message=message,result=result,results=results,
                sources=outcome.get('sources',{}),errors=errors,completed_sites=len(results),retrained=False)
    except Exception as exc:
        set_job(job_id, status='failed', message='Refresh could not complete. The previous saved forecast is unchanged.', error=str(exc))


def run_refresh(job_id,site=None,include_radar=True):
    """Compatibility entry point; every refresh is now the shared batch."""
    return run_refresh_all(job_id)


def start_refresh_all():
    with LOCK:
        # All callers share one batch, even older clients posting four sites.
        for job in JOBS.values():
            if job['status'] in {'queued', 'running'}:
                return dict(job)
        done = sorted((j for j in JOBS.values() if j['status'] in {'succeeded', 'partial','failed'}), key=lambda j: j['created_at'])
        for old in done[:-16]:
            JOBS.pop(old['job_id'], None)
        job_id = uuid.uuid4().hex
        JOBS[job_id] = {'job_id': job_id, 'site':'all','sites':['alta','vail','mansfield','jay'], 'include_radar':True,
                        'status': 'queued', 'message': 'Waiting for the current refresh to finish…',
                        'created_at': utc_now(), 'updated_at': utc_now(),'sources':{},'completed_sites':0}
        response = dict(JOBS[job_id])
        WORKER.submit(run_refresh_all,job_id)
        return response


def start_refresh(site,include_radar=False):
    if site not in SITES:raise ValueError('Choose one of the four supported mountains.')
    return start_refresh_all()


class Handler(BaseHTTPRequestHandler):
    server_version = 'SnowDashboard/1.0'

    def log_message(self, fmt, *args):
        print(f'[{utc_now()}] {fmt % args}', flush=True)

    def headers_for(self, code, content_type, length):
        self.send_response(code)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(length))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'self'; frame-ancestors 'none'")
        self.end_headers()

    def respond_json(self, data, code=200):
        raw = json.dumps(data, allow_nan=False, ensure_ascii=False).encode()
        self.headers_for(code, 'application/json; charset=utf-8', len(raw))
        self.wfile.write(raw)

    def host_allowed(self):
        host = self.headers.get('Host', '')
        return host in {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}

    def request_allowed(self):
        if not self.host_allowed():
            self.respond_json({'error': 'This dashboard accepts local browser requests only.'}, 403)
            return False
        origin = self.headers.get('Origin')
        if origin and origin not in {f'http://127.0.0.1:{self.server.server_port}', f'http://localhost:{self.server.server_port}'}:
            self.respond_json({'error': 'Cross-origin requests are disabled.'}, 403)
            return False
        return True

    def do_GET(self):
        if not self.request_allowed():
            return
        url = urlparse(self.path)
        try:
            if url.path == '/api/health':
                return self.respond_json({'status': 'ok', 'time': utc_now(), 'local_only': True})
            if url.path == '/api/mountains':
                from dashboard_forecasts import list_mountains
                return self.respond_json(list_mountains())
            if url.path == '/api/forecast':
                site = parse_qs(url.query).get('site', ['alta'])[0]
                if site not in SITES:
                    return self.respond_json({'error': 'Unknown mountain.'}, 400)
                return self.respond_json(read_forecast(site))
            if url.path.startswith('/api/jobs/'):
                key = url.path.removeprefix('/api/jobs/')
                with LOCK:
                    job = dict(JOBS[key]) if key in JOBS else None
                return self.respond_json(job if job else {'error': 'Refresh job not found.'}, 200 if job else 404)
            if url.path == '/api/jobs':
                with LOCK:
                    active = [{k: v for k, v in j.items() if k != 'result'} for j in JOBS.values() if j['status'] in {'running', 'queued'}]
                return self.respond_json(active)
            filename = {'/': 'index.html', '/index.html': 'index.html', '/app.js': 'app.js', '/styles.css': 'styles.css'}.get(url.path)
            if filename:
                file = STATIC / filename
            elif url.path.startswith('/reports/alta/') or url.path.startswith('/reports/weather/'):
                category, tail = url.path[len('/reports/'):].split('/', 1)
                directory = ROOT / 'outputs' / ('alta-nowcast' if category == 'alta' else 'nwp-weather')
                file = directory / (tail or 'report.html')
                if not file.resolve().is_relative_to(directory.resolve()) or file.suffix not in {'.html', '.png', '.pdf', '.csv', '.json'}:
                    return self.respond_json({'error': 'File not available.'}, 404)
            elif url.path == '/data/alta_nowcast/radar/model_scientific_audit.txt':
                file = ROOT / 'data/alta_nowcast/radar/model_scientific_audit.txt'
            else:
                return self.respond_json({'error': 'Not found.'}, 404)
            if not file.is_file():
                return self.respond_json({'error': 'File not found.'}, 404)
            raw = file.read_bytes()
            content_type = mimetypes.guess_type(file.name)[0] or 'application/octet-stream'
            self.headers_for(200, content_type, len(raw))
            self.wfile.write(raw)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            self.respond_json({'error': str(exc)}, 500)

    def do_POST(self):
        if not self.request_allowed():
            return
        endpoint=urlparse(self.path).path
        if endpoint not in {'/api/refresh','/api/refresh-all'}:
            return self.respond_json({'error': 'Not found.'}, 404)
        if self.headers.get('Content-Type', '').split(';', 1)[0] != 'application/json':
            return self.respond_json({'error': 'Use a JSON request.'}, 415)
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 2048:
                return self.respond_json({'error': 'Invalid request size.'}, 413)
            data = json.loads(self.rfile.read(length))
            if endpoint=='/api/refresh-all':
                if not isinstance(data,dict) or data:
                    raise ValueError('Refresh all expects an empty JSON object.')
                return self.respond_json(start_refresh_all(),202)
            if not isinstance(data, dict) or set(data) - {'site', 'include_radar'}:
                raise ValueError('Unsupported refresh options.')
            if not isinstance(data.get('site'), str) or not isinstance(data.get('include_radar', False), bool):
                raise ValueError('Invalid mountain or radar option.')
            return self.respond_json(start_refresh(data['site'], data.get('include_radar', False)), 202)
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            return self.respond_json({'error': str(exc)}, 400)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--open', action='store_true', help='Open the dashboard in your default browser')
    args = parser.parse_args()
    try:
        server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    except OSError as exc:
        if exc.errno == errno.EADDRINUSE and args.open:
            url = f'http://127.0.0.1:{args.port}/'
            try:
                with urlopen(url + 'api/health', timeout=2) as response:
                    health = json.load(response)
                if health.get('status') == 'ok' and health.get('local_only') is True:
                    webbrowser.open(url)
                    print(f'The snow dashboard is already running: {url}', flush=True)
                    return
            except Exception:
                pass
        raise SystemExit(f'Could not start the local dashboard on port {args.port}: {exc}') from exc
    url = f'http://127.0.0.1:{server.server_port}/'
    print(f'Snow dashboard: {url}\nRefresh runs saved models with new online inputs; it does not retrain.\nPress Ctrl+C to stop.', flush=True)
    if args.open:
        threading.Timer(.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        WORKER.shutdown(wait=False, cancel_futures=True)


if __name__ == '__main__':
    main()
