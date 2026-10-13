"""Authorization before the first test byte is read, bound to a frozen plan.

The full test-file hash commits its unknown ID order before any new access.
After real approval, hydrate the V3 ID policy deterministically from those
committed bytes. This initial authorized read has its own persisted receipt;
do not claim the ID-list policy existed before the authorized hydration.
"""
import copy
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

def sha_bytes(raw): return hashlib.sha256(raw).hexdigest()
def read_pinned(path, expected):
    p=Path(path)
    if not p.is_absolute() or not isinstance(expected,str) or len(expected)!=64:
        raise ValueError('Absolute path and external SHA256 required')
    raw=p.read_bytes()
    if sha_bytes(raw)!=expected:raise ValueError('Pinned document differs')
    return json.loads(raw)

def utc(text):
    if not isinstance(text,str):raise PermissionError('Explicit UTC timestamp required')
    at=datetime.fromisoformat(text.replace('Z','+00:00'))
    if at.tzinfo is None or at.utcoffset().total_seconds()!=0:raise PermissionError('UTC timestamp required')
    return at

def authorize(plan_path, plan_sha, authorization_path, authorization_sha):
    plan=read_pinned(plan_path,plan_sha)
    if authorization_path is None or authorization_sha is None:
        raise PermissionError('Separate real human final-test authorization required')
    auth=read_pinned(authorization_path,authorization_sha)
    at=utc(auth.get('authorized_at_utc'));freeze=utc(plan['frozen_at_utc'])
    if at<freeze or at>datetime.now(timezone.utc):raise PermissionError('Authorization must follow protocol freeze and not be future-dated')
    scope=auth.get('scope')
    if (auth.get('authorized') is not True or auth.get('plan_sha256')!=plan_sha
            or auth.get('test_sha256')!=plan['test_source']['sha256']
            or not isinstance(auth.get('authorized_by'),str) or not auth['authorized_by'].strip()
            or not isinstance(auth.get('source_user_message'),str) or not auth['source_user_message'].strip()
            or not isinstance(scope,dict) or scope.get('purpose')!='final_test'
            or scope.get('run_id')!=plan['run_id'] or scope.get('model_keys')!=plan['model_keys']):
        raise PermissionError('Human authorization does not cover this exact plan, source and run')
    return plan,auth

def save_new(path,value):
    p=Path(path)
    if p.exists():raise ValueError('Preserve existing authorization/access evidence')
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def hydrate_test_policy(plan_path,plan_sha,authorization_path,authorization_sha,template_path,template_sha,out):
    """Recheck pinned human authorization internally before any test access."""
    plan,auth=authorize(plan_path,plan_sha,authorization_path,authorization_sha)
    policy=read_pinned(template_path,template_sha)
    out=Path(out)
    if not out.is_dir():raise ValueError('Existing fresh receipt directory required')
    for name in ['INITIAL_AUTHORIZED_TEST_READ.json','HYDRATED_SOURCE_POLICY.json','DERIVED_SCOPE_AUTHORIZATION.json']:
        if (out/name).exists():raise ValueError('Preserve existing access evidence; no repeated hydration')
    test=plan['test_source'];p=Path(test['path'])
    if not p.is_absolute():raise ValueError('Absolute committed test source required')
    # Receipt is written immediately after the actual byte read, even if later
    # hash/schema validation fails. No model is loaded before this completes.
    raw=p.read_bytes();actual=sha_bytes(raw)
    save_new(out/'INITIAL_AUTHORIZED_TEST_READ.json',{'path':str(p),'actual_sha256':actual,
        'expected_sha256':test['sha256'],'plan_sha256':plan_sha,'authorization_sha256':authorization_sha,
        'read_at_utc':datetime.now(timezone.utc).isoformat(),'purpose':'authorized deterministic policy hydration',
        'source_user_message':auth['source_user_message'],'test_read':True})
    if actual!=test['sha256']:raise ValueError('Committed test bytes differ')
    rows=json.loads(raw)
    if not isinstance(rows,list) or len(rows)!=test['samples'] or not all(isinstance(r,dict) for r in rows):raise ValueError('Test row count/schema differs')
    if any(type(r.get('previously_inspected_test')) is not bool for r in rows):raise ValueError('Real prior-inspection flags required')
    if sum(r['previously_inspected_test'] for r in rows)!=test['previously_inspected_count']:raise ValueError('Prior-inspection accounting differs; no guessing or inference starts')
    policy['splits']['test']={'path':str(p),'sha256':test['sha256'],'samples':test['samples'],
        'sample_ids':[r.get('sample_id') for r in rows],'allowed_periods':test['allowed_periods'],'source_split':'test'}
    from split_target_contract import validate_policy
    validate_policy(policy)
    policy.update(protocol_plan_sha256=plan_sha,protocol_frozen_at_utc=plan['frozen_at_utc'],
                  id_policy_materialized_at_utc=datetime.now(timezone.utc).isoformat(),
                  id_policy_derivation='Deterministic ID order from the whole-file SHA committed before first authorized read; no data-informed configuration selection')
    path=out/'HYDRATED_SOURCE_POLICY.json';save_new(path,policy);digest=sha_bytes(path.read_bytes())
    derived=copy.deepcopy(auth);derived.update(policy_sha256=digest,derived_from_plan_sha256=plan_sha,
        derivation_note='Original real user approval covers this exact committed file and deterministic policy hydration; human message/time unchanged')
    derived_path=out/'DERIVED_SCOPE_AUTHORIZATION.json';save_new(derived_path,derived)
    return path,digest,derived_path,sha_bytes(derived_path.read_bytes())
