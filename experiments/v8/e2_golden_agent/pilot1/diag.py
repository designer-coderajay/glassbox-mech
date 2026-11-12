import sys
sys.path[:0] = ['.', 'experiments/v8/e2_golden_agent']
import system as s
m = s.HFAgentModel('Qwen/Qwen2.5-7B-Instruct', 'a09a35458c702b33eeacc393d103063234e8bc28', 'cuda', 'bfloat16')
x = s.System(m, 'e2-diag', 3)
dev = {q['id']: q for q in s.load_questions('dev')}
cases = [('e_1', 'persona'), ('e_2', 'persona'), ('e_1', 'answer_policy'),
         ('b_5120', 'invoice_tool_desc'), ('ci_6011', 'refund_tool_desc'),
         ('d_ridgeline', 'runbook_order'), ('d_ridgeline', None)]
for qid, f in cases:
    cfg = dict(s.BASE)
    if f:
        cfg[f] = s.CANDIDATE[f]
    print('=====', qid, 'only', f or 'baseline')
    for i in range(3):
        e = x.episode(cfg, dev[qid], i)
        print(e['label'], '| calls:', e['calls'], '| limit:', e['hit_limit'], '| parse_fail:', e['parse_failures'])
        print('   ', e['final'][:300].replace('\n', ' '))
