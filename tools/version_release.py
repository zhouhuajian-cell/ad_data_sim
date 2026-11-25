#!/usr/bin/env python3
import argparse,datetime,fcntl,json,os,tempfile
P='/opt/ue5_synth/dashboard/release_versions.json'
def main():
 a=argparse.ArgumentParser();a.add_argument('--component',required=True,choices=['platform','tsr','obj','gnd']);a.add_argument('--part',required=True,choices=['major','minor','patch']);a.add_argument('--note',required=True);o=a.parse_args()
 if not o.note.strip():raise ValueError('release note required')
 with open(P+'.lock','w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  with open(P,encoding='utf8') as f:d=json.load(f)
  v=[int(x) for x in d['versions'][o.component].split('.')];i={'major':0,'minor':1,'patch':2}[o.part];v[i]+=1
  for j in range(i+1,3):v[j]=0
  new='.'.join(map(str,v));d['versions'][o.component]=new;d.setdefault('changes',[]).append({'date':datetime.datetime.now().astimezone().isoformat(timespec='seconds'),'component':o.component,'version':new,'summary':o.note.strip()})
  fd,tmp=tempfile.mkstemp(prefix='.version_',dir=os.path.dirname(P))
  try:
   with os.fdopen(fd,'w',encoding='utf8') as f:json.dump(d,f,ensure_ascii=False,indent=2)
   os.replace(tmp,P)
  finally:
   if os.path.exists(tmp):os.unlink(tmp)
  print(o.component,new)
if __name__=='__main__':main()
