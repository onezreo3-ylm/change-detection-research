import io
import struct
import unittest
import zlib

from scripts.second_release_quarantine import CheckedInflater,local_payload_start
from scripts.second_release_chunked_quarantine import chunk_ranges,remaining_wall


class TrainingContainerStagingTests(unittest.TestCase):
    def compressed(self,data):
        obj=zlib.compressobj(wbits=-15);return obj.compress(data)+obj.flush()

    def test_train_container_stream_exact_bytes_crc_and_hash(self):
        raw=b'Rar!\x1a\x07\x01\x00'+b'opaque synthetic container'*1000;compressed=self.compressed(raw);out=io.BytesIO()
        decoder=CheckedInflater(out,len(raw),f'{zlib.crc32(raw):08x}')
        for i in range(0,len(compressed),7):decoder.consume(compressed[i:i+7])
        result=decoder.finish();self.assertEqual(out.getvalue(),raw);self.assertEqual(result['bytes'],len(raw))

    def test_unexpected_size_crc_or_container_type_never_certifies(self):
        for raw,size,crc in [(b'Rar!\x1a\x07\x00abc',3,'00000000'),(b'Rar!\x1a\x07\x00abc',10,'00000000'),
                             (b'not-a-rar',9,f'{zlib.crc32(b"not-a-rar"):08x}')]:
            with self.subTest(raw=raw),self.assertRaises(ValueError):
                decoder=CheckedInflater(io.BytesIO(),size,crc);decoder.consume(self.compressed(raw));decoder.finish()

    def test_only_exact_author_training_member_local_header_is_accepted(self):
        name=b'SECOND_train_set.rar';header=struct.pack('<4s5H3I2H',b'PK\x03\x04',20,8,8,0,0,0,0,0,len(name),0)+name
        entry=dict(name=name.decode(),flags=8,local_header_offset=0)
        self.assertEqual(local_payload_start(header,entry),30+len(name))
        with self.assertRaises(ValueError):local_payload_start(header,{**entry,'name':'SECOND_total_test.zip'})

    def test_fixed_ranges_cover_training_stream_once_without_gaps_or_overlap(self):
        rows=list(chunk_ranges(82,2406180274))
        self.assertEqual(len(rows),36)
        self.assertEqual(rows[0][0],82);self.assertEqual(rows[-1][1],82+2406180274-1)
        self.assertEqual(sum(z-a+1 for a,z in rows),2406180274)
        self.assertTrue(all(rows[i][1]+1==rows[i+1][0] for i in range(len(rows)-1)))
        self.assertTrue(all(z-a+1<=67108864 for a,z in rows))

    def test_transport_repair_does_not_reset_original_wall_budget(self):
        self.assertAlmostEqual(remaining_wall(7.688),1792.312)
        for value in (1800,1801,-1,float('nan')):
            with self.assertRaises(ValueError):remaining_wall(value)


if __name__=='__main__':unittest.main()
