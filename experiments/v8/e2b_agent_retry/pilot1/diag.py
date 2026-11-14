"""Dev-only diagnostic after E2b pilot 1 (PROTOCOL §9). Not part of any scored run."""
import sys
sys.path[:0] = ['.', 'experiments/v8/e2b_agent_retry']
import system as s
m = s.HFAgentModel('Qwen/Qwen2.5-7B-Instruct', 'a09a35458c702b33eeacc393d103063234e8bc28', 'cuda', 'bfloat16')
x = s.System(m, 'e2b-diag', 3)
dev = {q['id']: q for q in s.load_questions('dev')}
cases = [
    ('ci_4502', 'baseline', {}),
    ('ci_4502', 'only persona', {'persona': 'polite'}),
    ('cd_5703', 'only refund_fix', {'refund_fix': 'fixed'}),
    ('d_fellgate', 'only quote_example', {'quote_example': 'compute_then_apply'}),
    ('e_1', 'only answer_policy', {'answer_policy': 'own_knowledge'}),
    ('e_2', 'all_but eligibility_tool', {**s.CANDIDATE, 'eligibility_tool': 'registered'}),
]
for qid, name, change in cases:
    cfg = {**s.BASE, **change}
    q = dev[qid]
    print('=====', qid, name, '| accepted', q['accepted'], '| rejected', q['rejected'])
    for i in range(3):
        e = x.episode(cfg, q, i)
        print(e['label'], '| calls:', e['calls'], '| limit:', e['hit_limit'])
        print('   ', e['final'][:400].replace('\n', ' '))
