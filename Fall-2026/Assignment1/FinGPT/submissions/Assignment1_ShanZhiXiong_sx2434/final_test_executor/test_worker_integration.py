"""Synthetic-only parent/child control-flow checks; fake GPU backend, no real test.

Windows uses a test-only fcntl stand-in. Linux additionally uses the real lock.
"""
import copy, hashlib, json, os, sys, tempfile, types, unittest
from pathlib import Path
from datetime import datetime,timezone,timedelta
from unittest.mock import patch
try:import fcntl
except ImportError:
    sys.modules['fcntl']=types.SimpleNamespace(LOCK_EX=1,LOCK_NB=2,LOCK_UN=8,flock=lambda *a:None)
import final_test_worker as w
import authorization_gate as gate

TEXT='[Positive Developments]: Synthetic fact.\n[Potential Concerns]: Synthetic risk.\n[Prediction & Analysis]:\nPrediction: Up by 1-2%\nAnalysis: Synthetic explanation.'
class Tokenizer:
    chat_template=None;bos_token_id=1;eos_token_id=2
    def encode(self,*a,**kw):return [1,3]
    def decode(self,*a,**kw):return TEXT
class Course:
    def calc_metrics(self,a,b):return {'valid_count':len(a),'mse':0}
    def parse_answer(self,a):return {'prediction':1.5}
def summarize():pass
summarize.__globals__['course']=Course()

class Integration(unittest.TestCase):
    def setUp(self):
        root=Path(__file__).parent/'synthetic_scratch';root.mkdir(exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=root);self.r=Path(self.tmp.name).resolve();self.b=self.r/'bundle';self.b.mkdir();(self.r/'setup').mkdir();self.out=self.r/'output';self.calls=[];self.loaded=[]
        period='2024-03-03 to 2024-03-10';rows=[]
        for i in range(300):
            prompt='SYNTHETIC INPUT ONLY '+str(i);symbol='FAKE'+str(i)
            rows.append({'sample_id':hashlib.sha256((symbol+'|'+period+'|'+prompt).encode()).hexdigest(),'split':'test','source_split':'test','symbol':symbol,'period':period,'prompt':prompt,'answer':TEXT,'label':'Up by 1-2%','previously_inspected_test':i<20})
        self.test=self.r/'SYNTHETIC_TEST.json';self.write(self.test,rows)
        policy={'splits':{}}
        for split,start,end,letter in [('train','2024-02-18','2024-02-25','b'),('validation','2024-02-25','2024-03-03','c')]:
            policy['splits'][split]={'path':str(self.r/(split+'.json')),'sha256':'1'*64,'samples':1,'sample_ids':[letter*64],'allowed_periods':[start+' to '+end],'source_split':'train'}
        policy['known_test_sha256']=[w.digest(self.test)];self.write(self.b/'SOURCE_POLICY_REMOTE.json',policy)
        self.plan={'model_keys':['qwen','deepseek','llama'],'new_training':False,'run_id':'SYNTHETIC_INTEGRATION','frozen_at_utc':(datetime.now(timezone.utc)-timedelta(minutes=1)).isoformat(),'output_directory':str(self.out),'generation':{'context_tokens':6144,'max_new_tokens':1920,'do_sample':False,'prefill_chunk_tokens':512,'batch_sizes':{'qwen':16,'deepseek':16,'llama':8}},'dependency_sha256':{'SOURCE_POLICY_REMOTE.json':w.digest(self.b/'SOURCE_POLICY_REMOTE.json')},'test_source':{'path':str(self.test),'sha256':w.digest(self.test),'samples':300,'previously_inspected_count':20,'allowed_periods':[period]},'validated_prompt_max_tokens':dict.fromkeys(['qwen','deepseek','llama'],10)}
        self.authpath=self.r/'SYNTHETIC_AUTH.json';self.commit()
        self.registry={k:{'model_id':'SYNTHETIC/'+k,'cache':str(self.r/k)} for k in self.plan['model_keys']}
        self.bindings={k:{'selected_checkpoint':{'checkpoint':str(self.r/(k+'_adapter'))},'cache_files_sha256':{'tokenizer.json':'a'*64,'tokenizer_config.json':'b'*64}} for k in self.plan['model_keys']}
        owner=self
        class Engine:
            def __init__(self,key):
                owner.loaded.append(key);self.checkpoint=Path(owner.bindings[key]['selected_checkpoint']['checkpoint']);self.tok=Tokenizer();self.guard=21
                self.torch=types.SimpleNamespace(cuda=types.SimpleNamespace(get_device_properties=lambda n:types.SimpleNamespace(total_memory=24*2**30),empty_cache=lambda:None))
            def ids(self,row):return row['input_ids']
            def adapter(self):pass
            def generate(self,rows,*a):
                return [{**r,'generated':TEXT,'generated_token_ids':[2],'generated_token_sha256':gate.sha_bytes(json.dumps([2]).encode()),'generated_tokens':1,'ended_eos':True,'hit_generation_limit':False,'peak_allocated_gib':8,'peak_reserved_gib':9,'batch_size':len(rows)} for r in rows]
        self.backend=types.SimpleNamespace(Engine=Engine,status=lambda *a,**k:None,summarize=summarize)
    def tearDown(self):self.tmp.cleanup()
    def write(self,p,o):p.write_text(json.dumps(o));return w.digest(p)
    def commit(self):
        self.ph=self.write(self.b/'PLAN.json',self.plan)
        self.ah=self.write(self.authpath,{'authorized':True,'plan_sha256':self.ph,'test_sha256':self.plan['test_source']['sha256'],'authorized_at_utc':datetime.now(timezone.utc).isoformat(),'authorized_by':'SYNTHETIC ONLY','source_user_message':'Synthetic fixture, never human authorization','scope':{'purpose':'final_test','run_id':self.plan['run_id'],'model_keys':self.plan['model_keys']}})
    def preflight(self,plan,keys):return self.bindings,self.registry,{k:Tokenizer() for k in keys},self.backend,'SYNTHETIC SYSTEM'
    def spawn(self,command,**kwargs):
        self.calls.append(command[command.index('--child-model')+1]);self.assertEqual(kwargs['pass_fds'],(int(command[-1]),))
        with patch.object(sys,'argv',command[1:]),patch.object(os,'getppid',lambda:os.getpid()):w.main()
        return types.SimpleNamespace(returncode=0)
    def run_main(self,preflight=None):
        argv=['worker','--plan-sha',self.ph,'--authorization',str(self.authpath),'--authorization-sha',self.ah]
        with patch.object(w,'R',self.r),patch.object(w,'B',self.b),patch.object(w,'P',self.r/'core'),patch.object(w,'preflight',preflight or self.preflight),patch.object(w.subprocess,'check_output',lambda *a,**k:''),patch.object(w.subprocess,'run',self.spawn),patch.object(sys,'argv',argv):w.main()
    def test_parent_three_child_lifecycle_and_every_boundary_receipt(self):
        self.run_main();self.assertEqual(self.calls,['qwen','deepseek','llama']);self.assertEqual(self.loaded,self.calls)
        self.assertEqual(json.loads((self.out/'COMPLETED.json').read_text())['outputs'],1800)
        for key in self.calls:
            for family in ['base','adapter']:
                rows=[json.loads(l) for l in (self.out/key/(family+'_outputs.jsonl')).read_text().splitlines()]
                self.assertEqual(len(rows),300);self.assertEqual(sum(r['previously_inspected_test'] for r in rows),20)
                self.assertTrue(all(r['boundary_receipt']['sample_id']==r['sample_id'] for r in rows))
                if key=='deepseek' and family=='adapter':self.assertTrue(all(r['evaluation_variant']=='adapter_legacy_unclosed' for r in rows))
    def test_preflight_failure_precedes_test_access_and_output_creation(self):
        reads=[];original=Path.read_bytes
        def tracked(p):
            if p==self.test:reads.append(p)
            return original(p)
        def failed(*a):raise ValueError('Synthetic preflight failure')
        with patch.object(Path,'read_bytes',tracked):
            with self.assertRaises(ValueError):self.run_main(failed)
        self.assertEqual(reads,[]);self.assertFalse(self.out.exists());self.assertEqual(self.loaded,[])
    def test_over_envelope_uses_single_row_without_filtering(self):
        self.plan['validated_prompt_max_tokens']['llama']=1;self.commit();self.run_main()
        rows=[json.loads(l) for l in (self.out/'llama/base_outputs.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows),300);self.assertTrue(all(r['batch_size']==1 and r['single_row_batch_required'] for r in rows))
        self.assertEqual(self.calls,['qwen','deepseek','llama'])
    def test_hard_budget_aborts_before_all_children(self):
        with patch.object(Tokenizer,'encode',lambda *a,**k:list(range(4225))):
            with self.assertRaises(ValueError):self.run_main()
        self.assertTrue((self.out/'INITIAL_AUTHORIZED_TEST_READ.json').exists());self.assertTrue((self.out/'FAILED.json').exists());self.assertEqual(self.loaded,[]);self.assertEqual(self.calls,[])
    def test_mixed_length_partition_order(self):
        cohort=[{'sample_id':str(i),'single_row_batch_required':i==5} for i in range(20)]
        schedule=w.generation_groups(cohort,8)
        self.assertEqual([len(g) for g in schedule],[1]*8+[8,4])
        self.assertEqual([r['sample_id'] for g in schedule for r in g],[str(i) for i in range(20)])
        self.assertEqual(schedule,w.generation_groups(cohort,8))
    def test_optimized_interpreter_rejected_before_authorization(self):
        with patch.object(w.sys,'flags',types.SimpleNamespace(optimize=1)),patch.object(w,'authorize',side_effect=AssertionError('must not be called')):
            with self.assertRaises(ValueError):w.main()

if __name__=='__main__':unittest.main()
