"""Network-free checks of the public archive metadata boundary."""
import io
import unittest
import zipfile

from scripts.second_zip_directory import (DirectoryWindows,check_range,directory_bounds,
                                         inspect_directory,public_download_locator,smoke)
from scripts.second_zip_session_directory import first_range_total


class MetadataBoundaryTests(unittest.TestCase):
    def test_ignored_range_is_rejected_before_any_body_is_admitted(self):
        for status,headers in [(200,{'Content-Length':'999999'}),(206,{'Content-Range':'bytes0-9/100'}),
                               (206,{'Content-Range':'bytes 0-9/100','Content-Length':'11'})]:
            with self.assertRaises(ValueError):check_range(status,headers,0,9,100,100)
        self.assertEqual(check_range(206,{'Content-Range':'bytes 0-9/100','Content-Length':'10'},0,9,100,100),10)

    def test_sparse_directory_can_list_but_cannot_open_large_image_member(self):
        b=io.BytesIO()
        with zipfile.ZipFile(b,'w',compression=zipfile.ZIP_STORED) as z:z.writestr('test/opaque.png',b'x'*262144)
        data=b.getvalue();tail=data[-65558:];bounds=directory_bounds(tail,len(data));central=data[bounds['offset']:bounds['offset']+bounds['size']]
        rows=inspect_directory(len(data),tail,central,bounds)
        self.assertEqual(rows[0]['name'],'test/opaque.png')
        sparse=DirectoryWindows(len(data),[(bounds['offset'],central),(len(data)-len(tail),tail)])
        with zipfile.ZipFile(sparse) as z:
            with self.assertRaises(ValueError):z.read('test/opaque.png')

    def test_public_form_must_keep_author_file_id_and_exact_anonymous_domain(self):
        form='<form method="get" action="https://drive.usercontent.google.com/download">'+''.join(
            f'<input name="{k}" value="{v}">' for k,v in dict(id='allowed',export='download',confirm='t',uuid='fixture').items())+'</form>'
        self.assertIn('id=allowed',public_download_locator(form,'allowed'))
        with self.assertRaises(ValueError):public_download_locator(form,'other')
        with self.assertRaises(ValueError):public_download_locator(form.replace('drive.usercontent.google.com','example.com'),'allowed')

    def test_eocd_and_directory_bounds_fail_closed(self):
        with self.assertRaises(ValueError):directory_bounds(b'not a zip',1000)
        b=io.BytesIO()
        with zipfile.ZipFile(b,'w') as z:z.writestr('README.txt','fixture')
        data=b.getvalue();changed=bytearray(data);changed[-2:]=b'\x01\x00'
        with self.assertRaises(ValueError):directory_bounds(bytes(changed),len(changed))

    def test_directory_fixture_smoke_has_no_network_or_label_decode(self):
        value=smoke();self.assertEqual(value['network_requests'],0);self.assertFalse(value['image_or_label_decode'])

    def test_session_correction_requires_exact_partial_four_byte_header(self):
        self.assertEqual(first_range_total(206,{'Content-Range':'bytes 0-3/3500000000'}),3500000000)
        for status,header in [(200,'bytes 0-3/3500000000'),(206,'bytes 0-4/3500000000'),(206,'')]:
            with self.assertRaises(ValueError):first_range_total(status,{'Content-Range':header})


if __name__=='__main__':unittest.main()
