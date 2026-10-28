"""One anonymous-session correction for public directory metadata, no member reads."""
from pathlib import Path
from urllib.request import Request,build_opener,HTTPCookieProcessor
from urllib.parse import urlparse
from http.cookiejar import CookieJar
import hashlib
import json
import re

import yaml

from scripts.second_zip_directory import (ROOT,allowed_host,public_download_locator,check_range,
                                         directory_bounds,inspect_directory)


def first_range_total(status,headers):
    match=re.fullmatch(r'bytes 0-3/([0-9]+)',headers.get('Content-Range',''))
    if status!=206 or not match or int(match[1])<=131072:
        raise ValueError('Explicit four-byte public Range is unsupported; no body read or further retry')
    return int(match[1])


def run():
    path=ROOT/'configs/second_zip_directory_session_20261026.yaml';v=yaml.safe_load(path.read_text(encoding='utf-8'))
    if (v['maximum_http_requests']!=4 or v['maximum_warning_html_bytes']!=16384 or
        v['maximum_directory_bytes']!=4194304 or v['member_payload_reads'] is not False or
        v['additional_retry_if_failed'] is not False):raise ValueError('Frozen single-session scope changed')
    out=ROOT/v['output_dir']
    if out.exists():raise FileExistsError('Preserve prior session correction; no retry')
    out.mkdir();opener=build_opener(HTTPCookieProcessor(CookieJar()))
    record=dict(status='running',http_requests=0,application_archive_bytes_read=0,
        user_credentials_or_cookies_read=False,archive_fully_downloaded=False,member_payloads_opened=False,
        image_or_label_decode=False,additional_retry_if_failed=False)
    types={'application/zip','application/octet-stream','application/x-zip-compressed'}
    def open_once(url,start=None,end=None):
        if record['http_requests']>=4:raise ValueError('Frozen four-request cap reached')
        headers={'User-Agent':'Research-metadata-audit/1.0','Accept-Encoding':'identity'}
        if start is not None:headers['Range']=f'bytes={start}-{end}'
        record['http_requests']+=1
        response=opener.open(Request(url,headers=headers),timeout=v['timeout_seconds'])
        if not allowed_host(response.url):response.close();raise ValueError('Unexpected/authentication redirect; stop')
        return response
    def archive_range(url,start,end,total=None):
        with open_once(url,start,end) as response:
            record['last_response_metadata']=dict(status=response.status,host=urlparse(response.url).hostname,
                content_type=response.headers.get_content_type(),content_range=response.headers.get('Content-Range'),
                content_length=response.headers.get('Content-Length'))
            if response.headers.get_content_type() not in types:
                raise ValueError('Public session still returned non-archive content; no body read or further retry')
            if total is None:total=first_range_total(response.status,response.headers)
            count=check_range(response.status,response.headers,start,end,total,v['maximum_directory_bytes'])
            data=response.read(count+1)
            if len(data)!=count:raise ValueError('Incomplete/oversized archive metadata Range')
            record['application_archive_bytes_read']+=len(data)
            return data,total
    try:
        with open_once(v['public_warning_url']) as response:
            if response.headers.get_content_type()!='text/html':raise ValueError('Unexpected public warning type; body not read')
            declared=response.headers.get('Content-Length')
            if declared and int(declared)>16384:raise ValueError('Warning metadata too large')
            html=response.read(16385)
            if len(html)>16384:raise ValueError('Warning metadata too large')
        (out/'public_warning_page.html').write_bytes(html)
        url=public_download_locator(html.decode('utf-8'),v['file_id'])
        record['public_page_sha256']=hashlib.sha256(html).hexdigest()
        prefix,total=archive_range(url,0,3)
        if prefix!=b'PK\x03\x04':raise ValueError('Release does not begin with the expected ZIP local header')
        tail,_=archive_range(url,total-v['tail_bytes'],total-1,total);bounds=directory_bounds(tail,total)
        if bounds['size']>v['maximum_directory_bytes'] or bounds['entries']>v['maximum_entries']:
            raise ValueError('Directory exceeds frozen metadata cap')
        central,_=archive_range(url,bounds['offset'],bounds['offset']+bounds['size']-1,total)
        entries=inspect_directory(total,tail,central,bounds)
        (out/'tail.bin').write_bytes(tail);(out/'central_directory.bin').write_bytes(central)
        with (out/'entries.json').open('x',encoding='utf-8') as f:json.dump(entries,f,indent=2)
        record.update(status='directory_metadata_read_no_members_opened',bounds=bounds,entry_count=len(entries),
            possible_metadata_members=[e['name'] for e in entries if any(t in Path(e['name']).name.lower() for t in ('readme','license','licence','copying'))],
            tail_sha256=hashlib.sha256(tail).hexdigest(),central_directory_sha256=hashlib.sha256(central).hexdigest())
    except Exception as error:
        record.update(status='stopped_preserve_no_additional_endpoint_retry',error_type=type(error).__name__,reason=str(error))
    with (out/'review.json').open('x',encoding='utf-8') as f:json.dump(record,f,indent=2)
    return record


if __name__=='__main__':print(json.dumps(run()))
