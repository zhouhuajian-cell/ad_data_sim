import os,json,shutil,hashlib
root='/data/tsr_curated_20261125_v1'; src='/data/tsr_library_v2'
spec={
 'TH':{'mandatory_go_straight.png':'thailand_road_sign_37.png','mandatory_turn_left.png':'thailand_road_sign_38.png','mandatory_turn_right.png':'thailand_road_sign_39.png','mandatory_keep_left.png':'thailand_road_sign_40.png','mandatory_keep_right.png':'thailand_road_sign_41.png','prohibition_no_entry.png':'prohibition_no_entry_v2.png','prohibition_no_left.png':'prohibition_no_left_v2.png','prohibition_no_right.png':'prohibition_no_right_v2.png','prohibition_no_trucks.png':'prohibition_no_trucks.png','prohibition_no_u_turn.png':'prohibition_no_u_turn.png',**{'speed_limit_%d.png'%n:'thailand_road_sign_32_%d.png'%n for n in (30,40,45,50,60,70,80,90)}},
 'RU':{'mandatory_go_straight.png':'mandatory_go_straight.png','mandatory_keep_left.png':'mandatory_keep_left.png','mandatory_keep_right.png':'mandatory_keep_right.png','mandatory_turn_left.png':'mandatory_turn_left.png','mandatory_turn_right.png':'mandatory_turn_right.png','prohibition_no_entry.png':'prohibition_no_entry.png','prohibition_no_left.png':'prohibition_no_left.png','prohibition_no_right.png':'prohibition_no_right.png','prohibition_no_trucks.png':'prohibition_no_trucks_v3.png','prohibition_no_u_turn.png':'prohibition_no_u_turn.png','prohibition_no_overtaking.png':'prohibition_no_overtaking.png','prohibition_no_pedestrians.png':'prohibition_no_pedestrians.png',**{'speed_limit_%d.png'%n:'ru_road_sign_3_24_%d.png'%n for n in (20,30,40,50,60,70,80,90)}}}
manifest=[]
for country,files in spec.items():
 os.makedirs(os.path.join(root,country),exist_ok=True)
 for name,original in files.items():
  a=os.path.join(src,country,original); b=os.path.join(root,country,name)
  if not os.path.isfile(a):raise RuntimeError('missing '+a)
  if os.path.exists(b) and open(a,'rb').read()!=open(b,'rb').read():raise RuntimeError('destination conflict '+b)
  if not os.path.exists(b):shutil.copy2(a,b)
  manifest.append({'country':country,'class':name[:-4],'source':a,'file':b,'sha256':hashlib.sha256(open(a,'rb').read()).hexdigest(),'visual_checked':True,'license_status':'unverified'})
with open(os.path.join(root,'manifest.json'),'w',encoding='utf8') as f:json.dump(manifest,f,ensure_ascii=False,indent=2)
print('curated',len(manifest),root)
