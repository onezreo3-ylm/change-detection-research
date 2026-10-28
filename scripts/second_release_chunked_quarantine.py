"""One fixed-chunk repair with cumulative cost; same author archive, no chunk retry."""
from pathlib import Path
from urllib.request import Request,build_opener,HTTPCookieProcessor
from http.cookiejar import CookieJar
from datetime import datetime,timezone
import hashlib
import json
import math
import os
import shutil
import sys
import time

import yaml

from scripts.second_zip_directory import ROOT,allowed_host,public_download_locator,check_range
from scripts.second_release_quarantine import CheckedInflater,local_payload_start
from scripts.windows_shared_json import atomic_json as atomic
from scripts.levir_tail_risk_supervision import creation_ticks


def chunk_ranges(start,size,chunk=67108864):
    if start<0 or size<=0 or chunk<=0:raise ValueError('Positive bounded sequential range plan required')
    for offset in range(0,size,chunk):yield start+offset,start+min(size,offset+chunk)-1


def remaining_wall(previous,limit=1800):
    if not math.isfinite(previous) or not 0<=previous<limit:raise ValueError('Original cumulative staging budget exhausted')
    return limit-previous


def run():
    config=ROOT/'configs/second_release_chunked_quarantine_20261026.yaml';v=yaml.safe_load(config.read_text(encoding='utf-8'))
    original=yaml.safe_load((ROOT/v['reference_staging_config']).read_text(encoding='utf-8'))
    old=json.loads((ROOT/v['previous_failure']).read_text(encoding='utf-8'))
    if (v['range_bytes']!=67108864 or v['maximum_cumulative_http_requests']!=44 or v['whole_cumulative_wall_seconds']!=1800 or
        v['retry_any_chunk'] is not False or v['raw_test_member_open'] is not False or
        old['status']!='failed_preserve_partial_no_retry' or old['train_container_bytes_written']!=0):
        raise ValueError('Only the documented zero-payload failure correction is admitted')
    allowance=remaining_wall(old['elapsed_seconds']);out=ROOT/v['output_dir']
    if out.exists():raise FileExistsError('One chunked correction already exists; no further transfer retry')
    if shutil.disk_usage(ROOT).free<original['member_uncompressed_bytes']+original['minimum_extra_free_bytes']:
        raise RuntimeError('Insufficient quarantine disk')
    review=json.loads((ROOT/original['directory_review']).read_text(encoding='utf-8'))
    entries=json.loads((ROOT/original['directory_entries']).read_text(encoding='utf-8'))
    entry=next(e for e in entries if e['name']==original['member_name'])
    if (entry['compressed_bytes']!=original['member_compressed_bytes'] or entry['uncompressed_bytes']!=original['member_uncompressed_bytes'] or
        entry['crc32']!=original['member_crc32'] or entry['local_header_offset']!=0 or
        review['bounds']['total_bytes']!=original['outer_archive_bytes']):raise ValueError('Author training-member identity changed')
    out.mkdir();started=time.monotonic();opener=build_opener(HTTPCookieProcessor(CookieJar()))
    state=dict(status='running',started_utc=datetime.now(timezone.utc).isoformat(),http_requests=0,
        previous_http_requests=old['http_requests'],archive_bytes_read=0,previous_archive_bytes_read=old['archive_bytes_read'],
        previous_wall_seconds=old['elapsed_seconds'],train_container_bytes_written=0,completed_chunks=0,
        test_archive_opened=False,rar_extracted=False,image_or_label_decode=False,new_dataset_admitted=False,
        model_load_or_inference=False,automatic_retry=False)
    atomic(out/'process_identity.json',dict(pid=os.getpid(),creation_ticks=creation_ticks(os.getpid()),
        executable=sys.executable,command_line=sys.argv,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        config_sha256=hashlib.sha256(config.read_bytes()).hexdigest()))
    def heartbeat():
        elapsed=time.monotonic()-started
        if elapsed>allowance:raise TimeoutError('Original cumulative1800second staging cap reached')
        atomic(out/'progress.json',dict(state,elapsed_seconds=elapsed,cumulative_wall_seconds=old['elapsed_seconds']+elapsed,
            updated_utc=datetime.now(timezone.utc).isoformat()))
    def open_once(url,start=None,end=None):
        heartbeat()
        if state['http_requests']+old['http_requests']>=44:raise ValueError('Cumulative request cap reached')
        headers={'User-Agent':'Research-metadata-audit/1.0','Accept-Encoding':'identity'}
        if start is not None:headers['Range']=f'bytes={start}-{end}'
        state['http_requests']+=1
        response=opener.open(Request(url,headers=headers),timeout=original['per_read_timeout_seconds'])
        if not allowed_host(response.url):response.close();raise ValueError('Unexpected/authentication redirect; stop')
        return response
    def archive_response(url,start,end):
        response=open_once(url,start,end)
        try:
            state['last_response_metadata']=dict(status=response.status,content_type=response.headers.get_content_type(),
                content_range=response.headers.get('Content-Range'),content_length=response.headers.get('Content-Length'))
            if response.headers.get_content_type() not in ('application/zip','application/octet-stream','application/x-zip-compressed'):
                size=response.headers.get('Content-Length')
                if response.headers.get_content_type()=='text/html' and size and int(size)<=v['error_html_limit_bytes']:
                    page=response.read(v['error_html_limit_bytes']+1)
                    if len(page)<=v['error_html_limit_bytes']:(out/'public_download_error.html').write_bytes(page)
                raise ValueError('Fixed chunk returned non-archive type; no chunk retry')
            check_range(response.status,response.headers,start,end,original['outer_archive_bytes'],v['range_bytes'])
            return response
        except BaseException:response.close();raise
    try:
        with open_once(original['warning_url']) as response:
            if response.headers.get_content_type()!='text/html':raise ValueError('Unexpected public warning type')
            html=response.read(16385)
            if len(html)>16384:raise ValueError('Public warning metadata cap exceeded')
        url=public_download_locator(html.decode('utf-8'),original['file_id'])
        with archive_response(url,0,4095) as response:header=response.read(4097)
        if len(header)!=4096:raise ValueError('Training local-header length mismatch')
        state['archive_bytes_read']+=len(header);(out/'local_header.bin').write_bytes(header)
        b=review['bounds']
        with archive_response(url,b['offset'],b['offset']+b['size']-1) as response:central=response.read(b['size']+1)
        state['archive_bytes_read']+=len(central)
        if hashlib.sha256(central).hexdigest()!=review['central_directory_sha256']:
            raise ValueError('Author directory changed; no substitute resource')
        start=local_payload_start(header,entry);ranges=list(chunk_ranges(start,entry['compressed_bytes'],v['range_bytes']))
        if 3+len(ranges)+old['http_requests']>44 or ranges[-1][1]>=min(e['local_header_offset'] for e in entries if e['name']=='SECOND_total_test.zip'):
            raise ValueError('Frozen train-only request/range plan is invalid')
        state['planned_chunks']=len(ranges);compressed_sha=hashlib.sha256();partial=out/'SECOND_train_set.rar.partial'
        with partial.open('xb') as writer:
            decoder=CheckedInflater(writer,entry['uncompressed_bytes'],entry['crc32']);last=0.
            for a,z in ranges:
                with archive_response(url,a,z) as response:
                    remaining=z-a+1
                    while remaining:
                        if time.monotonic()-started>allowance:raise TimeoutError('Original cumulative staging cap reached')
                        block=response.read(min(1048576,remaining))
                        if not block:raise ValueError('Incomplete fixed chunk; no retry')
                        remaining-=len(block);state['archive_bytes_read']+=len(block);compressed_sha.update(block);decoder.consume(block)
                        state['train_container_bytes_written']=decoder.size
                        if time.monotonic()-last>=5:heartbeat();last=time.monotonic()
                state['completed_chunks']+=1;heartbeat()
            member=decoder.finish();writer.flush();os.fsync(writer.fileno())
        target=out/'SECOND_train_set.rar'
        if partial.resolve().parent!=out.resolve() or target.resolve().parent!=out.resolve() or target.exists():
            raise ValueError('Unexpected quarantine rename path')
        partial.rename(target)
        state.update(status='completed_same_author_train_container_quarantined_not_admitted',member=member,
            compressed_member_sha256=compressed_sha.hexdigest(),requested_range=dict(start=ranges[0][0],end=ranges[-1][1]),
            source_directory_sha256=review['central_directory_sha256'])
    except BaseException as error:state.update(status='failed_preserve_no_further_same_archive_transfer',error_type=type(error).__name__,reason=str(error))
    state.update(elapsed_seconds=time.monotonic()-started,cumulative_wall_seconds=old['elapsed_seconds']+time.monotonic()-started,
        cumulative_http_requests=old['http_requests']+state['http_requests'],cumulative_archive_bytes_read=old['archive_bytes_read']+state['archive_bytes_read'])
    atomic(out/'status.json',state);atomic(out/'progress.json',dict(state,updated_utc=datetime.now(timezone.utc).isoformat()))
    return state


if __name__=='__main__':
    r=run();print(json.dumps(r));raise SystemExit(0 if r['status'].startswith('completed') else 1)
