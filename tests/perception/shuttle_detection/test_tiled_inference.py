"""CPU tests for non-oracle image tiling, model adapter and global NMS."""
import numpy as np
import pytest
from src.perception.shuttle_detection.tiled_inference import (
    generate_tiles, map_and_filter, predict_tiled, class_aware_nms, TileDetection,
)


def test_cover_entire_frame_with_overlap_and_last_edge():
    tiles=generate_tiles(width=1280,height=720,tile_size=640,overlap=0.25)
    assert len(tiles)==6
    assert sorted(set(t.left for t in tiles))==[0,480,640]
    assert sorted(set(t.top for t in tiles))==[0,80]
    cover=np.zeros((720,1280),dtype=np.uint8)
    for t in tiles:cover[t.top:t.bottom,t.left:t.right]=1
    assert cover.all()
    assert all(t.right<=1280 and t.bottom<=720 for t in tiles)


def test_small_image_one_tile_no_invalid_padding():
    tiles=generate_tiles(200,100,640,0.25)
    assert [(t.left,t.top,t.right,t.bottom) for t in tiles]==[(0,0,200,100)]


def test_invalid_config_rejected():
    for kwargs in [dict(width=100,height=100,tile_size=0,overlap=0.2),
                   dict(width=100,height=100,tile_size=32,overlap=1),
                   dict(width=100,height=100,tile_size=32,overlap=-.01)]:
        with pytest.raises(ValueError): generate_tiles(**kwargs)


def test_map_and_filter_remaps_clips_and_rejects_invalid():
    t=generate_tiles(1280,720,640,0.25)[-1]
    assert (t.left,t.top)==(640,80)
    items=map_and_filter([(20,10,30,20,0.9,0),(600,620,630,640,0.5,0),
                           (1,1,1,5,0.8,0)],t,width=1280,height=720)
    assert len(items)==2
    assert items[0].xyxy==(660,90,670,100)
    assert items[1].xyxy==(1240,700,1270,720)


def test_global_nms_deduplicates_across_overlap_and_preserves_other_class():
    boxes=[TileDetection((100,100,120,120),.9,0),
           TileDetection((101,101,121,121),.8,0),
           TileDetection((100,100,120,120),.7,1)]
    result=class_aware_nms(boxes,iou_threshold=.5,max_det=10)
    assert [(r.confidence,r.class_id) for r in result]==[(.9,0),(.7,1)]


class FakeBoxes:
    def __init__(self, rows):
        arr=np.array(rows,dtype=float).reshape(-1,6)
        self.xyxy=arr[:,:4];self.conf=arr[:,4];self.cls=arr[:,5]
    def __len__(self):return len(self.conf)


class FakeResult:
    def __init__(self,rows):self.boxes=FakeBoxes(rows)


class FakeModel:
    def __init__(self):self.calls=[]
    def predict(self,source,**kwargs):
        self.calls.append((source.shape,kwargs))
        idx=len(self.calls)
        if idx==1:return [FakeResult([[490,100,510,112,.65,0]])]
        if idx==2:return [FakeResult([[10,100,30,112,.8,0]])]
        return [FakeResult([])]


def test_inference_adapter_remaps_and_dedups_without_ground_truth_roi():
    model=FakeModel()
    image=np.zeros((720,1280,3),np.uint8)
    boxes,meta=predict_tiled(model,image,tile_size=640,overlap=.25,
                             imgsz=640,conf=.01,iou=.45,device='cpu',max_det=64)
    assert meta['mode']=='tiled' and meta['tiles']==6
    assert meta['candidate_count']==2 and meta['kept_count']==1
    assert meta['uses_gt_roi'] is False
    assert boxes[0].xyxy==(490,100,510,112)
    assert boxes[0].confidence==.8
    assert len(model.calls)==6
    assert all(k['imgsz']==640 for _,k in model.calls)


def test_no_detections_returns_structured_empty_not_fake_box():
    model=FakeModel()
    model.predict=lambda source,**kwargs:[FakeResult([])]
    boxes,meta=predict_tiled(model,np.zeros((20,30,3),np.uint8),tile_size=64,
                             overlap=0.2,imgsz=64,conf=.01,iou=.45,device='cpu',max_det=10)
    assert boxes==[] and meta['tiles']==1 and meta['candidate_count']==0
