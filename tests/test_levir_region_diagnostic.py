import unittest
import numpy as np

from scripts.diagnose_levir_regions import eligible, region_records, select_names, stitch


class LevirRegionDiagnosticTests(unittest.TestCase):
    def test_reconstruction_merges_one_region_across_original_tile_seam(self):
        mask=np.zeros((16,16),dtype=np.uint8)
        mask[4:6,6:10]=1
        merged=stitch([(x,y,mask[y:y+8,x:x+8]) for y in [0,8] for x in [0,8]],16,8)
        records=region_records(merged,{"perfect":mask},connectivity=8,tile_size=8)
        self.assertEqual(len(records),1)
        self.assertEqual(records[0]["area"],8)
        self.assertTrue(records[0]["touches_tile_grid"])

    def test_missing_and_duplicate_tile_coverage_both_rejected(self):
        tile=np.zeros((8,8),dtype=np.uint8)
        valid=[(x,y,tile) for y in [0,8] for x in [0,8]]
        for tiles in [valid[:-1],valid+[valid[0]]]:
            with self.assertRaises(ValueError):stitch(tiles,16,8)

    def test_unaligned_origin_rejected(self):
        with self.assertRaises(ValueError):stitch([(1,0,np.zeros((8,8)))],16,8)

    def test_four_and_eight_connectivity_are_reported_distinctly(self):
        mask=np.eye(3,dtype=np.uint8)
        self.assertEqual(len(region_records(mask,{},connectivity=4,tile_size=3)),3)
        self.assertEqual(len(region_records(mask,{},connectivity=8,tile_size=3)),1)

    def test_zero_overlap_is_region_specific_even_when_tile_has_tp(self):
        mask=np.zeros((16,16),dtype=np.uint8)
        mask[2:4,2:4]=1
        mask[10:12,10:12]=1
        pred=np.zeros_like(mask)
        pred[2:4,2:4]=1
        rows=region_records(mask,{"model":pred},connectivity=8,tile_size=16)
        self.assertEqual([r["overlap_pixels"]["model"] for r in rows],[4,0])

    def test_primary_filter_rejects_source_edge_and_grid_contact(self):
        cfg={"primary_min_area":4,"primary_max_area":20,"primary_exclude_source_edge":True,"primary_exclude_tile_grid":True}
        base={"area":8,"touches_source_edge":False,"touches_tile_grid":False}
        self.assertTrue(eligible(base,cfg))
        for change in [{"area":3},{"area":21},{"touches_source_edge":True},{"touches_tile_grid":True}]:
            self.assertFalse(eligible(dict(base,**change),cfg))

    def test_empty_label_with_false_prediction_does_not_create_true_region(self):
        self.assertEqual(region_records(np.zeros((8,8)),{"fp":np.ones((8,8))},connectivity=8,tile_size=8),[])

    def test_source_selection_is_order_independent_and_has_no_score_input(self):
        names=["a.png","b.png","c.png","d.png"]
        self.assertEqual(select_names(names,2,"fixed"),select_names(list(reversed(names)),2,"fixed"))
        with self.assertRaises(ValueError):select_names(names+[names[0]],2,"fixed")

    def test_nonbinary_or_unaligned_predictions_rejected(self):
        with self.assertRaises(ValueError):region_records(np.zeros((8,8)),{"bad":np.zeros((4,4))},connectivity=8,tile_size=8)
        with self.assertRaises(ValueError):region_records(np.full((8,8),2),{},connectivity=8,tile_size=8)


if __name__=="__main__":unittest.main()
