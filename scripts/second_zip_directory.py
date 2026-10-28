"""Read only an author's public ZIP directory; refuse ignored/oversized ranges."""
from __future__ import annotations

import argparse
import hashlib
from html.parser import HTMLParser
import io
import json
from pathlib import Path
import re
import struct
from urllib.parse import urlencode,urlparse
from urllib.request import Request,urlopen
import zipfile

import yaml

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'configs/second_zip_directory_20261026.yaml'


class PublicForm(HTMLParser):
    def __init__(self):super().__init__();self.action=None;self.method=None;self.fields={};self.inside=False
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if tag=='form':
            if self.action is not None:raise ValueError('Ambiguous public release forms')
            self.action=a.get('action');self.method=a.get('method','get').lower();self.inside=True
        if tag=='input' and self.inside and a.get('name'):
            if a['name'] in self.fields:raise ValueError('Duplicate public metadata form field')
            self.fields[a['name']]=a.get('value','')
    def handle_endtag(self,tag):
        if tag=='form':self.inside=False


def public_download_locator(html,file_id):
    p=PublicForm();p.feed(html)
    u=urlparse(p.action or '')
    if (p.method!='get' or u.scheme!='https' or u.hostname!='drive.usercontent.google.com' or
        u.path!='/download' or u.username or u.password or u.port not in (None,443) or
        p.fields.get('id')!=file_id or p.fields.get('export')!='download' or
        set(p.fields)!={'id','export','confirm','uuid'}):
        raise ValueError('Not the single author-linked anonymous public-download form')
    return p.action+'?'+urlencode(p.fields)


def check_range(status,headers,start,end,total,maximum):
    expected=end-start+1
    content_range=headers.get('Content-Range','')
    if (status!=206 or content_range!=f'bytes {start}-{end}/{total}' or expected>maximum or
        headers.get('Content-Length') is not None and int(headers['Content-Length'])!=expected):
        raise ValueError('Server ignored/mismatched Range; archive body must not be read')
    return expected


def directory_bounds(tail,total):
    marker=tail.rfind(b'PK\x05\x06')
    if marker<0 or len(tail)-marker<22:raise ValueError('Missing complete ZIP EOCD')
    fields=struct.unpack_from('<4s4H2IH',tail,marker)
    _,disk,central_disk,disk_entries,entries,size,offset,comment=fields
    if marker+22+comment!=len(tail) or disk or central_disk or disk_entries!=entries:
        raise ValueError('Ambiguous/multidisk archive metadata')
    if entries==65535 or size==0xffffffff or offset==0xffffffff:
        raise ValueError('ZIP64 requires a separately reviewed metadata adapter; no fallback')
    location=total-len(tail)+marker
    if not entries or offset+size>location:raise ValueError('Invalid central-directory bounds')
    return dict(entries=entries,size=size,offset=offset,total_bytes=total)


class DirectoryWindows(io.RawIOBase):
    """Sparse in-memory file: any attempted member-payload read raises."""
    def __init__(self,total,windows):self.total,self.windows,self.position=total,windows,0
    def seekable(self):return True
    def readable(self):return True
    def tell(self):return self.position
    def seek(self,offset,whence=0):
        pos=offset if whence==0 else self.position+offset if whence==1 else self.total+offset if whence==2 else -1
        if not 0<=pos<=self.total:raise ValueError('Sparse metadata seek outside archive')
        self.position=pos;return pos
    def read(self,size=-1):
        if size<0:size=self.total-self.position
        size=min(size,self.total-self.position);chunks=[]
        while size:
            found=False
            for start,data in self.windows:
                if start<=self.position<start+len(data):
                    n=min(size,start+len(data)-self.position)
                    chunks.append(data[self.position-start:self.position-start+n]);self.position+=n;size-=n;found=True;break
            if not found:raise ValueError('Member payload is outside admitted metadata windows')
        return b''.join(chunks)


def inspect_directory(total,tail,central,bounds):
    sparse=DirectoryWindows(total,[(bounds['offset'],central),(total-len(tail),tail)])
    with zipfile.ZipFile(sparse) as archive:
        info=archive.infolist()
        if len(info)!=bounds['entries']:raise ValueError('Directory entry count mismatch')
        return [dict(name=i.orig_filename,compressed_bytes=i.compress_size,uncompressed_bytes=i.file_size,
                     compression=i.compress_type,crc32=f'{i.CRC:08x}',local_header_offset=i.header_offset,
                     flags=i.flag_bits,is_directory=i.is_dir()) for i in info]


def allowed_host(url):
    p=urlparse(url)
    return p.scheme=='https' and (p.hostname=='drive.google.com' or p.hostname=='drive.usercontent.google.com' or
                                  (p.hostname or '').endswith('.googleusercontent.com'))


def run():
    v=yaml.safe_load(CONFIG.read_text(encoding='utf-8'));out=ROOT/v['output_dir']
    if out.exists():raise FileExistsError('Preserve existing directory attempt; no retry')
    if (v['maximum_http_requests']!=3 or v['tail_bytes']!=131072 or v['maximum_directory_bytes']!=4194304 or
        v['member_payload_reads'] is not False or v['image_or_label_decode'] is not False or v['archive_full_download'] is not False):
        raise ValueError('Frozen metadata-only scope changed')
    page=(ROOT/v['release_html']).read_bytes();review=json.loads((ROOT/v['release_review']).read_text(encoding='utf-8'))
    if hashlib.sha256(page).hexdigest()!=review['page_sha256']:raise ValueError('Original public page changed')
    url=public_download_locator(page.decode('utf-8'),v['file_id']);out.mkdir()
    record=dict(status='running',http_requests=0,application_archive_bytes_read=0,archive_fully_downloaded=False,
        member_payloads_opened=False,image_or_label_decode=False,locator_sha256=hashlib.sha256(url.encode()).hexdigest())
    def request(method,start=None,end=None,total=None):
        if record['http_requests']>=v['maximum_http_requests']:raise ValueError('Metadata request cap reached')
        headers={'User-Agent':'Research-metadata-audit/1.0','Accept-Encoding':'identity'}
        if start is not None:headers['Range']=f'bytes={start}-{end}'
        record['http_requests']+=1
        response=urlopen(Request(url,method=method,headers=headers),timeout=v['timeout_seconds'])
        try:
            if not allowed_host(response.url):raise ValueError('Unexpected release redirect/authentication host')
            if response.headers.get_content_type() not in v['allowed_content_types']:
                raise ValueError('Archive endpoint did not return admitted archive type; body not read')
            if method=='HEAD':
                length=response.headers.get('Content-Length')
                if response.status!=200 or length is None or int(length)<=v['tail_bytes']:
                    raise ValueError('Exact archive length unavailable; no range/body read')
                return dict(total_bytes=int(length),etag=response.headers.get('ETag'),last_modified=response.headers.get('Last-Modified'))
            count=check_range(response.status,response.headers,start,end,total,v['maximum_directory_bytes'])
            data=response.read(count+1)
            if len(data)!=count:raise ValueError('Incomplete/oversized metadata range')
            record['application_archive_bytes_read']+=len(data)
            return data
        finally:response.close()
    try:
        identity=request('HEAD');record['published_identity_headers']=identity;total=identity['total_bytes']
        tail=request('GET',total-v['tail_bytes'],total-1,total);bounds=directory_bounds(tail,total)
        if bounds['size']>v['maximum_directory_bytes'] or bounds['entries']>v['maximum_entries']:
            raise ValueError('Archive directory exceeds frozen metadata cap')
        central=request('GET',bounds['offset'],bounds['offset']+bounds['size']-1,total)
        entries=inspect_directory(total,tail,central,bounds)
        (out/'tail.bin').write_bytes(tail);(out/'central_directory.bin').write_bytes(central)
        with (out/'entries.json').open('x',encoding='utf-8') as f:json.dump(entries,f,indent=2)
        text_files=[e['name'] for e in entries if any(t in Path(e['name']).name.lower() for t in ('readme','license','licence','copying'))]
        record.update(status='directory_metadata_read_no_members_opened',bounds=bounds,entry_count=len(entries),
            possible_metadata_members=text_files,tail_sha256=hashlib.sha256(tail).hexdigest(),
            central_directory_sha256=hashlib.sha256(central).hexdigest())
    except Exception as error:record.update(status='stopped_preserve_metadata_failure',error_type=type(error).__name__,reason=str(error))
    with (out/'review.json').open('x',encoding='utf-8') as f:json.dump(record,f,indent=2)
    return record


def smoke():
    # A large opaque member ensures the directory reader cannot reach payloads.
    b=io.BytesIO()
    with zipfile.ZipFile(b,'w',compression=zipfile.ZIP_STORED) as z:
        z.writestr('train/image1/fixture.png',b'x'*262144);z.writestr('README.txt',b'fixture metadata')
    data=b.getvalue();tail=data[-65558:];bounds=directory_bounds(tail,len(data))
    rows=inspect_directory(len(data),tail,data[bounds['offset']:bounds['offset']+bounds['size']],bounds)
    if len(rows)!=2:raise ValueError('Synthetic directory smoke failed')
    sparse=DirectoryWindows(len(data),[(bounds['offset'],data[bounds['offset']:]),(len(data)-len(tail),tail)])
    try:
        sparse.seek(0);sparse.read(1)
    except ValueError:pass
    else:raise ValueError('Synthetic payload denial failed')
    return dict(status='passed_synthetic_directory_only_smoke',image_or_label_decode=False,network_requests=0)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['smoke','run']);a=p.parse_args()
    print(json.dumps(smoke() if a.action=='smoke' else run()))


if __name__=='__main__':main()
