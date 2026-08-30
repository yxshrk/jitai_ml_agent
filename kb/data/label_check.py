import csv, collections
D = 'kuairand-starter-kit/KuaiRand-Pure/data'
vdur = {}
with open(f'{D}/video_features_basic_pure.csv') as fh:
    for r in csv.DictReader(fh): vdur[r['video_id']] = r['video_duration']
dates = collections.Counter(); n = 0
exc = collections.Counter(); ex_samples = []
agree_log = agree_feat = 0
for f in ('log_standard_4_08_to_4_21_pure.csv', 'log_standard_4_22_to_5_08_pure.csv'):
    with open(f'{D}/{f}') as fh:
        for r in csv.DictReader(fh):
            d = int(r['date']); dates[d] += 1
            if d > 20220428: continue
            n += 1
            y = int(r['long_view']); p = int(r['play_time_ms']); dl = int(r['duration_ms'])
            rule_log = int(p >= min(dl, 18000))
            df = vdur.get(r['video_id'], '')
            rule_feat = int(p >= min(int(float(df)), 18000)) if df not in ('', 'NaN') else rule_log
            agree_log += (y == rule_log); agree_feat += (y == rule_feat)
            if y != rule_log:
                kind = ('label=1 but play<thr' if y == 1 else 'label=0 but play>=thr')
                exc[(kind, 'short<=18s' if dl <= 18000 else 'long>18s', 'dur_ms==0' if dl == 0 else 'dur>0')] += 1
                if len(ex_samples) < 8: ex_samples.append((y, p, dl, df, r['is_click']))
print('rows with date 20220408:', dates.get(20220408, 0), '| first date in files:', min(dates), '| last:', max(dates))
print(f'agreement with rule using log duration_ms : {agree_log/n:.4%}')
print(f'agreement with rule using video_duration  : {agree_feat/n:.4%}')
print('exception breakdown (kind, video length, duration_ms zero?):')
for k, v in exc.most_common(): print('  ', k, v)
print('samples (long_view, play_time_ms, duration_ms, video_duration_feat, is_click):')
for s in ex_samples: print('  ', s)
