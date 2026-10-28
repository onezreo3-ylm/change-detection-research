"""Stage only the author-linked train RAR; no image decode, extraction or model."""
from pathlib import Path
from urllib.request import Request,build_opener,HTTPCookieProcessor
from http.cookiejar import CookieJar
from datetime import datetime,timezone
import hashlib
import io
import json
import os
import shutil
import struct
import sys
import time
import zlib

import yaml

from scripts.second_zip_directory import ROOT,allowed_host,public_download_locator,check_range
from scripts.windows_shared_json import atomic_json as atomic
from scripts.levir_tail_risk_supervision import creation_ticks


class CheckedInflater:
    def __init__(self,writer,expected_size,expected_crc):
        self.writer,self.expected_size,self.expected_crc=writer,expected_size,expected_crc
        self.decoder=zlib.decompressobj(-15);self.size=0;self.crc=0;self.digest=hashlib.sha256();self.prefix=bytearray()
    def emit(self,data):
        if self.size+len(data)>self.expected_size:raise ValueError('Training member expands beyond declared size')
        if len(self.prefix)<8:self.prefix.extend(data[:8-len(self.prefix)])
        self.writer.write(data);self.size+=len(data);self.crc=zlib.crc32(data,self.crc);self.digest.update(data)
    def consume(self,data):
        while data:
            self.emit(self.decoder.decompress(data,1048576))
            if self.decoder.unused_data:raise ValueError('Compressed member range contains trailing data')
            data=self.decoder.unconsumed_tail
    def finish(self):
        if not self.decoder.eof:raise ValueError('Incomplete raw ZIP deflate stream')
        self.emit(self.decoder.flush())
        if self.size!=self.expected_size or f'{self.crc & 0xffffffff:08x}'!=self.expected_crc:
            raise ValueError('Complete training member length/CRC mismatch')
        if not (bytes(self.prefix).startswith(b'Rar!\x1a\x07\x00') or bytes(self.prefix).startswith(b'Rar!\x1a\x07\x01\x00')):
            raise ValueError('Staged training member is not the declared RAR container')
        return dict(bytes=self.size,crc32=f'{self.crc & 0xffffffff:08x}',sha256=self.digest.hexdigest())


def local_payload_start(header,entry):
    if len(header)<30:raise ValueError('Incomplete ZIP local header')
    sig,version,flags,method,mtime,mdate,crc,csize,usize,nlen,xlen=struct.unpack_from('<4s5H3I2H',header)
    if sig!=b'PK\x03\x04' or method!=8 or flags!=entry['flags'] or len(header)<30+nlen+xlen:
        raise ValueError('Training local header differs from directory')
    name=header[30:30+nlen].decode('utf-8' if flags & 2048 else 'cp437')
    if name!=entry['name']:raise ValueError('Local filename differs from the fixed training member')
    return entry['local_header_offset']+30+nlen+xlen


def run():
    cfg=ROOT/'configs/second_release_quarantine_20261026.yaml';v=yaml.safe_load(cfg.read_text(encoding='utf-8'))
    if (v['maximum_http_requests']!=4 or v['whole_wall_seconds']!=1800 or v['extract_members'] is not False or
        v['image_or_label_decode'] is not False or v['test_archive_open'] is not False or v['automatic_retry'] is not False):
        raise ValueError('Frozen quarantine scope changed')
    out=ROOT/v['output_dir']
    if out.exists():raise FileExistsError('Existing staging attempt retained; no automatic retry')
    review=json.loads((ROOT/v['directory_review']).read_text(encoding='utf-8'))
    entries=json.loads((ROOT/v['directory_entries']).read_text(encoding='utf-8'))
    matches=[e for e in entries if e['name']==v['member_name']]
    if len(matches)!=1:raise ValueError('Unique author-linked training member required')
    entry=matches[0]
    if (entry['compressed_bytes']!=v['member_compressed_bytes'] or entry['uncompressed_bytes']!=v['member_uncompressed_bytes'] or
        entry['crc32']!=v['member_crc32'] or entry['compression']!=8 or entry['local_header_offset']!=0 or
        review['bounds']['total_bytes']!=v['outer_archive_bytes']):raise ValueError('Fixed archive/member identity differs')
    if shutil.disk_usage(ROOT).free<v['member_uncompressed_bytes']+v['minimum_extra_free_bytes']:
        raise RuntimeError('Insufficient staging disk; no smaller dataset substitution')
    out.mkdir();started=time.monotonic();opener=build_opener(HTTPCookieProcessor(CookieJar()))
    state=dict(status='running',started_utc=datetime.now(timezone.utc).isoformat(),http_requests=0,archive_bytes_read=0,
        train_container_bytes_written=0,test_archive_opened=False,rar_extracted=False,image_or_label_decode=False,
        new_dataset_admitted=False,model_load_or_inference=False,automatic_retry=False)
    atomic(out/'process_identity.json',dict(pid=os.getpid(),creation_ticks=creation_ticks(os.getpid()),
        executable=sys.executable,command_line=sys.argv,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        config_sha256=hashlib.sha256(cfg.read_bytes()).hexdigest()))
    def heartbeat():
        elapsed=time.monotonic()-started
        if elapsed>v['whole_wall_seconds']:raise TimeoutError('Whole staging wall cap reached')
        atomic(out/'progress.json',dict(state,elapsed_seconds=elapsed,updated_utc=datetime.now(timezone.utc).isoformat()))
    def open_once(url,start=None,end=None):
        heartbeat()
        if state['http_requests']>=4:raise ValueError('Frozen request cap reached')
        headers={'User-Agent':'Research-metadata-audit/1.0','Accept-Encoding':'identity'}
        if start is not None:headers['Range']=f'bytes={start}-{end}'
        state['http_requests']+=1
        response=opener.open(Request(url,headers=headers),timeout=v['per_read_timeout_seconds'])
        if not allowed_host(response.url):response.close();raise ValueError('Unexpected/authenticated redirect; stop')
        return response
    def range_response(url,start,end):
        response=open_once(url,start,end)
        try:
            if response.headers.get_content_type() not in ('application/zip','application/octet-stream','application/x-zip-compressed'):
                raise ValueError('Training archive Range returned non-archive type; no body read')
            check_range(response.status,response.headers,start,end,v['outer_archive_bytes'],v['member_compressed_bytes']+8192)
            return response
        except BaseException:response.close();raise
    try:
        with open_once(v['warning_url']) as response:
            if response.headers.get_content_type()!='text/html':raise ValueError('Unexpected public warning type; body not read')
            html=response.read(16385)
            if len(html)>16384:raise ValueError('Warning HTML cap exceeded')
        url=public_download_locator(html.decode('utf-8'),v['file_id'])
        with range_response(url,0,4095) as response:header=response.read(4097)
        if len(header)!=4096:raise ValueError('Local-header Range length mismatch')
        state['archive_bytes_read']+=len(header)
        bounds=review['bounds']
        with range_response(url,bounds['offset'],bounds['offset']+bounds['size']-1) as response:central=response.read(bounds['size']+1)
        state['archive_bytes_read']+=len(central)
        if hashlib.sha256(central).hexdigest()!=review['central_directory_sha256']:
            raise ValueError('Published archive directory changed before training-only staging')
        start=local_payload_start(header,entry);end=start+entry['compressed_bytes']-1
        test_start=min(e['local_header_offset'] for e in entries if e['name']=='SECOND_total_test.zip')
        if end>=test_start:raise ValueError('Staging range would cross into the sealed new test member')
        partial=out/'SECOND_train_set.rar.partial';target=out/'SECOND_train_set.rar';compressed_sha=hashlib.sha256()
        with partial.open('xb') as writer,range_response(url,start,end) as response:
            decoder=CheckedInflater(writer,entry['uncompressed_bytes'],entry['crc32']);remaining=entry['compressed_bytes'];last=0.
            while remaining:
                if time.monotonic()-started>v['whole_wall_seconds']:raise TimeoutError('Staging wall cap reached')
                block=response.read(min(v['chunk_bytes'],remaining))
                if not block:raise ValueError('Incomplete training-only member transfer')
                remaining-=len(block);state['archive_bytes_read']+=len(block);compressed_sha.update(block);decoder.consume(block)
                state['train_container_bytes_written']=decoder.size
                if time.monotonic()-last>=5:heartbeat();last=time.monotonic()
            member=decoder.finish();writer.flush();os.fsync(writer.fileno())
        if partial.resolve().parent!=out.resolve() or target.resolve().parent!=out.resolve() or target.exists():
            raise ValueError('Unexpected quarantine rename target')
        partial.rename(target)
        state.update(status='completed_training_container_quarantined_not_admitted',member=member,
            compressed_member_sha256=compressed_sha.hexdigest(),requested_range=dict(start=start,end=end),
            original_directory_sha256=review['central_directory_sha256'])
    except BaseException as error:
        state.update(status='failed_preserve_partial_no_retry',error_type=type(error).__name__,reason=str(error))
    state['elapsed_seconds']=time.monotonic()-started
    atomic(out/'status.json',state);atomic(out/'progress.json',dict(state,updated_utc=datetime.now(timezone.utc).isoformat()))
    return state


if __name__=='__main__':
    result=run();print(json.dumps(result));raise SystemExit(0 if result['status'].startswith('completed') else 1)
