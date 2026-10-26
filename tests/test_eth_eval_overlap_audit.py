import csv,hashlib,json,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from audit_eth_eval_overlap import audit_eval_overlap,run,EvalAuditBlocked


TRAIN_FIELDS=['image','sha256','role','source','split','is_positive','is_negative','n_boxes','location','in_official_training']
EVAL_FIELDS=['source_id','role','sha256','bytes','origin']


def write_csv(path,fields,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    return path


def setup(tmp_path):
    train=[
      {'image':'/data/pos.jpg','sha256':'A'*64,'role':'official_positives','source':'eth_main','split':'train','is_positive':'True','is_negative':'False','n_boxes':'1','location':'cab_1','in_official_training':'True'},
      {'image':'/data/neg.jpg','sha256':'B'*64,'role':'hard_negative','source':'hard_negative','split':'train','is_positive':'False','is_negative':'True','n_boxes':'0','location':'cab_1','in_official_training':'True'},
      {'image':'/data/contradiction.jpg','sha256':'C'*64,'role':'official_positives','source':'eth_main','split':'train','is_positive':'False','is_negative':'True','n_boxes':'0','location':'ticino_1','in_official_training':'True'},
    ]
    ev=[
      {'source_id':'neg','role':'FROZEN_EVAL','sha256':'B'*64,'bytes':'100','origin':'hard_negatives'},
      {'source_id':'synthetic','role':'FROZEN_EVAL','sha256':'D'*64,'bytes':'100','origin':'synthetic_3d'},
    ]
    val=[{'image':'/data/images/train/val.jpg','sha256':'E'*64,'role':'official_positives','source':'eth_main','split':'train','is_positive':'True','is_negative':'False','n_boxes':'1','location':'glc_2','in_official_training':'True'}]
    return (write_csv(tmp_path/'train.csv',TRAIN_FIELDS,train),
            write_csv(tmp_path/'eval.csv',EVAL_FIELDS,ev),
            write_csv(tmp_path/'val.csv',TRAIN_FIELDS,val))


def test_overlap_by_exact_sha_and_roles(tmp_path):
    train,ev,val=setup(tmp_path)
    r=audit_eval_overlap(train,ev,val)
    assert r['sha_overlap']['unique_sha256']==1
    assert r['sha_overlap']['training_rows']==1
    assert r['sha_overlap']['train_rows_by_role']=={'hard_negative':1}
    assert r['sha_overlap']['eval_rows_by_origin']=={'hard_negatives':1}


def test_contradictory_official_positive_is_reported(tmp_path):
    train,ev,val=setup(tmp_path)
    r=audit_eval_overlap(train,ev,val)
    assert len(r['contradictory_official_positive_rows'])==1
    assert r['contradictory_official_positive_rows'][0]['image'].endswith('contradiction.jpg')


def test_val_semantics_flags_train_split_and_train_path(tmp_path):
    train,ev,val=setup(tmp_path)
    r=audit_eval_overlap(train,ev,val)
    assert r['val_semantics']['split_values']=={'train':1}
    assert r['val_semantics']['in_official_training_true']==1
    assert r['val_semantics']['images_train_path_tokens']==1
    assert r['policy']['recall_or_false_positive_metrics_authorized'] is False


def test_read_only_source_manifests_and_versioned_outputs(tmp_path):
    train,ev,val=setup(tmp_path)
    before=[hashlib.sha256(p.read_bytes()).hexdigest() for p in (train,ev,val)]
    out=run(train,ev,tmp_path/'audit',val)
    after=[hashlib.sha256(p.read_bytes()).hexdigest() for p in (train,ev,val)]
    assert before==after
    assert (tmp_path/'audit'/'audit.json').is_file()
    assert (tmp_path/'audit'/'training_rows_overlapping_frozen_eval.csv').is_file()
    assert out['policy']['source_manifests_modified'] is False


def test_empty_manifest_blocks(tmp_path):
    train,ev,val=setup(tmp_path);empty=tmp_path/'empty.csv';empty.write_text('')
    with pytest.raises(EvalAuditBlocked,match='missing_or_empty'):
        audit_eval_overlap(empty,ev,val)


def test_existing_output_dir_refused(tmp_path):
    train,ev,val=setup(tmp_path);dest=tmp_path/'audit';dest.mkdir()
    with pytest.raises(EvalAuditBlocked,match='already_exists'):
        run(train,ev,dest,val)


def test_no_overlap_is_not_enough_to_call_eval_independent(tmp_path):
    train,ev,val=setup(tmp_path)
    rows=list(csv.DictReader(ev.open()));rows[0]['sha256']='F'*64
    write_csv(ev,EVAL_FIELDS,rows)
    r=audit_eval_overlap(train,ev,val)
    assert r['sha_overlap']['unique_sha256']==0
    assert r['status']=='AUDIT_ONLY_REVIEW_REQUIRED'
    assert r['policy']['new_independent_eval_set_created'] is False


def test_windows_path_separator_is_normalized(tmp_path):
    train,ev,val=setup(tmp_path)
    rows=list(csv.DictReader(val.open()))
    rows[0]['image']=r'C:\data\images\train\val.jpg'
    write_csv(val,TRAIN_FIELDS,rows)
    r=audit_eval_overlap(train,ev,val)
    assert r['val_semantics']['images_train_path_tokens']==1
