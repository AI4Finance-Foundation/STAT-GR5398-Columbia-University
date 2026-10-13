"""Synthetic fixtures only, all scratch on D:. Never reads the real test."""
import copy, json, unittest, tempfile
from pathlib import Path
from datetime import datetime, timezone, timedelta
from unittest.mock import patch
import authorization_gate as gate

B=Path(__file__).resolve().parent

class GateTests(unittest.TestCase):
    def setUp(self):
        scratch=B/'synthetic_scratch';scratch.mkdir(exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=scratch);self.root=Path(self.temp.name)
        self.test=self.root/'SYNTHETIC_TEST.json'
        row={'sample_id':'a'*64,'previously_inspected_test':True}
        self.test.write_text(json.dumps([row]));self.test_sha=gate.sha_bytes(self.test.read_bytes())
        self.plan={'frozen_at_utc':(datetime.now(timezone.utc)-timedelta(hours=1)).isoformat(),'run_id':'SYNTHETIC_ONLY','model_keys':['qwen','deepseek','llama'],'test_source':{'path':str(self.test),'sha256':self.test_sha,'samples':1,'previously_inspected_count':1,'allowed_periods':['2024-03-03 to 2024-03-10']}}
        self.pp=self.root/'PLAN.json';self.ph=self.write(self.pp,self.plan)
        self.auth={'authorized':True,'plan_sha256':self.ph,'test_sha256':self.test_sha,'authorized_at_utc':datetime.now(timezone.utc).isoformat(),'authorized_by':'SYNTHETIC FIXTURE','source_user_message':'Synthetic fixture, never real approval','scope':{'purpose':'final_test','run_id':'SYNTHETIC_ONLY','model_keys':self.plan['model_keys']}}
        self.ap=self.root/'AUTH_SYNTHETIC.json';self.ah=self.write(self.ap,self.auth)
        self.policy={'splits':{'train':{'path':str(self.root/'FAKE_TRAIN.json'),'sha256':'1'*64,'sample_ids':['b'*64],'samples':1,'allowed_periods':['2024-02-18 to 2024-02-25'],'source_split':'train'},'validation':{'path':str(self.root/'FAKE_VAL.json'),'sha256':'2'*64,'sample_ids':['c'*64],'samples':1,'allowed_periods':['2024-02-25 to 2024-03-03'],'source_split':'train'}},'known_test_sha256':[self.test_sha]}
        self.tp=self.root/'TEMPLATE.json';self.th=self.write(self.tp,self.policy)
        self.out=self.root/'receipts';self.out.mkdir()
    def tearDown(self):self.temp.cleanup()
    def write(self,p,obj):p.write_text(json.dumps(obj));return gate.sha_bytes(p.read_bytes())
    def hydrate(self):return gate.hydrate_test_policy(self.pp,self.ph,self.ap,self.ah,self.tp,self.th,self.out)
    def assert_reject_before_access(self):
        original=Path.read_bytes;access=[]
        def tracked(p):
            if p==self.test:access.append(p)
            return original(p)
        with patch.object(Path,'read_bytes',tracked):
            with self.assertRaises((PermissionError,ValueError,FileNotFoundError,KeyError)):self.hydrate()
        self.assertEqual(access,[])
    def test_missing_auth(self):self.ap=None;self.ah=None;self.assert_reject_before_access()
    def test_missing_file(self):self.ap=self.root/'ABSENT';self.assert_reject_before_access()
    def test_changed_auth_bytes(self):self.ap.write_text('{}');self.assert_reject_before_access()
    def test_changed_plan_bytes(self):self.pp.write_text('{}');self.assert_reject_before_access()
    def test_bad_template_pin(self):self.th='0'*64;self.assert_reject_before_access()
    def test_existing_receipt(self):(self.out/'INITIAL_AUTHORIZED_TEST_READ.json').write_text('{}');self.assert_reject_before_access()
    def test_mismatched_human_scope(self):
        mutations=[('authorized',False),('plan_sha256','0'*64),('test_sha256','0'*64),('authorized_by',''),('source_user_message',''),('authorized_at_utc',(datetime.now(timezone.utc)-timedelta(days=1)).isoformat()),('authorized_at_utc',(datetime.now(timezone.utc)+timedelta(days=1)).isoformat()),('authorized_at_utc','2026-01-01T00:00:00'),('scope',{'purpose':'validation','run_id':'SYNTHETIC_ONLY','model_keys':self.plan['model_keys']}),('scope',{'purpose':'final_test','run_id':'WRONG','model_keys':self.plan['model_keys']}),('scope',{'purpose':'final_test','run_id':'SYNTHETIC_ONLY','model_keys':['llama']})]
        for key,value in mutations:
            with self.subTest(key=key,value=value):
                a=copy.deepcopy(self.auth);a[key]=value;self.ah=self.write(self.ap,a);self.assert_reject_before_access()
    def test_valid_hydration_preserves_human_message_and_time(self):
        pp,ph,ap,ah=self.hydrate();p=gate.read_pinned(pp,ph);a=gate.read_pinned(ap,ah)
        self.assertEqual(p['splits']['test']['sample_ids'],['a'*64]);self.assertEqual(a['source_user_message'],self.auth['source_user_message']);self.assertEqual(a['authorized_at_utc'],self.auth['authorized_at_utc'])
        self.assertTrue((self.out/'INITIAL_AUTHORIZED_TEST_READ.json').exists())
        self.assert_reject_before_access()
    def test_hash_failure_retains_actual_read_receipt(self):
        self.test.write_text('[]')
        with self.assertRaises(ValueError):self.hydrate()
        self.assertTrue((self.out/'INITIAL_AUTHORIZED_TEST_READ.json').exists())
        self.assertFalse((self.out/'HYDRATED_SOURCE_POLICY.json').exists())
    def test_exposure_failure_retains_actual_read_receipt(self):
        self.plan['test_source']['previously_inspected_count']=0;self.ph=self.write(self.pp,self.plan);self.auth['plan_sha256']=self.ph;self.ah=self.write(self.ap,self.auth)
        with self.assertRaises(ValueError):self.hydrate()
        self.assertTrue((self.out/'INITIAL_AUTHORIZED_TEST_READ.json').exists())

if __name__=='__main__':unittest.main()
