import pymongo
import base64
from datetime import datetime, timedelta

def db_search(collection,user_id):
    
    c = collection['patient']
    info = c.find_one({'id':user_id})
    if info is not None:
        return {
            'name': info['name'],
            'dob': info['date of birth'],
            'sex': info['sex']
        }
    return None

def db_count(collection,user_id):
    
    c = collection['history']
    # first remove too old records(1 year from now)
    one_year_ago = (datetime.now() - timedelta(days = 365)).strftime('%Y-%m-%d')
    query = {'time': {'$lt':one_year_ago}}
    try:
        result = c.delete_many(query)
        print(f"Documents deleted: {result.deleted_count}")
    except Exception as e:
        print(f"An error occurred: {e}")
        
    # compute over other records
    info = c.find({'id':user_id}).sort('time')
    
    avg_sugar,w1 = 0,0.5
    avg_pressure,w2 = 0,0.7
    
    for _,record in enumerate(info):
        avg_sugar = avg_sugar*w1 + float(record.get('blood sugar',0))
        avg_pressure = avg_pressure*w2 + float(record.get('blood pressure',0)) 
    
    health_score = (avg_sugar + avg_pressure)/2 
    
    return health_score
    
    
def daily_record(collection,user_id,time):
    
    c = collection['history']
    info = c.find_one({'id':user_id,'time':time})
    print(info)
    if info is not None:
        
        return {
            'sport': info['sports'],
            'bloodsugar': info['blood sugar'],
            'bloodpressure': info['blood pressure']
        }
        
    return None

def update_record(collection, query, data):
    
    c = collection['history']
    record = c.find(query)
    # if today has no record, insert
    kmap = {'sports':'sports','blood_sugar': 'blood sugar', 'blood_pressure': 'blood pressure'}
    pdata = {kmap[k]:data[k] for k in data.keys()}
    if list(record) == []:
        for qk in query.keys():
            pdata[qk] = query[qk]
        c.insert_one(pdata)
    # already has record, update
    else:
        c.update_one(query,{'$set':pdata})
    

def find_user(collection,id,time):
    c = collection['history']
    data = c.find_one({'id':id,'time':time})
    ans = dict()
    for k, v in data.items():
        if k == "blood pressure":
            ans["bloodpressure"] = v
        elif k == "blood sugar":
            ans["bloodsugar"] = v
        elif k != "_id":
            ans[k] = v
        
        print(ans)
    return ans

def decode_session(x):
    
    id = base64.b64decode(x)
    id = id.decode("utf-8")
    
    return id

coded = 'Vm0wd2QyUXlVWGxWV0d4V1YwZDRWMVl3WkRSV01WbDNXa1JTVjAxV2JETlhhMUpUVjBaS2RHVkdXbFppVkZaeVZteFZlRll5VGtsalJtaG9UVmhDVVZkV1pEUlRNazE0V2toR1VtSkdXbGhaYTJoRFZWWmtWMXBFVWxSTmF6RTBWMnRvUjFWdFNrZFhiR2hhWVRKb1JGWldXbUZqVmtaMFVteFNUbUY2UlRGV1ZFb3dWakZhV0ZOcmFHaFNlbXhXVm1wT1QwMHhjRlpYYlVaclVqQTFSMWRyV2xOVWJVcEdZMFZ3VjJKVVJYZFdha1pYWkVaT2MxZHNhR2xTYTNCWlYxWmtNR1F5VW5OalJtUllZbFZhY2xWcVJtRlRSbGw1VFZSU1ZrMXJjRWxhU0hCSFZqSkZlVlZZWkZwbGEzQklXWHBHVDJSV1ZuTlhiV2hzWWxob2IxWnRNWGRVTWtsNVVtdGthbEp0VWxsWmJGWmhZMnhXYzFWclpGZGlSbkJaV2xWYVQxWlhTbFpYVkVwV1lrWktTRlpxU2tabFZsWlpXa1prYUdFeGNGbFhhMVpoVkRKTmVGcElUbWhTTW5oVVZGY3hiMWRzV1hoWGJYUk9VbTE0V0ZaWGRHdFdNV1JJWVVac1dtSkhhRlJXTUZwVFZqRndSMVJ0ZUdsU2JYY3hWa1phVTFVeFduSk5XRXBxVWxkNGFGVXdhRU5TUmxweFUydGFiRlpzU2xwWlZWcHJWVEZLVjJOSWJGZFdSVXBvVmtSS1QyUkdTbkphUm1ocFZqTm9WVmRXVWs5Uk1XUkhWMjVTVGxaRlNsaFVWbVEwVjBaYVdHUkhkRmhTTUhCSlZsZDRjMWR0U2tkWGJXaGFUVzVvV0ZsNlJsZGpiSEJIV2tkc1UySnJTbUZXTW5oWFdWWlJlRmRzYUZSaVJuQlpWbXRXZDFZeGJISlhhM1JUVW14d2VGVldhRzloTVZwelYycEdWMDF1YUhKWlZXUkdaV3hHY21KR1pHbFhSVXBKVm10U1MxVXhXWGhYYmxaVllrZG9jRlpxU205bGJHUllaVWM1YVUxcmJEUldNalZUVkd4a1NGVnNXbFZXTTFKNlZHeGFWMlJIVWtoa1JtaFRUVVpaTVZac1pEUmpNV1IwVTJ0b2FGSnNTbGhVVmxwM1YwWnJlRmRyZEdwaVZrcElWbGQ0YTJGV1NuUlBWRTVYVFc1b1dGbHFTa1psUm1SWldrVTFWMVpzY0ZWWFZsSkhaREZaZUdKSVNsaGhNMUpVVlcxNGQyVkdWWGxrUjBacFVteHdlbFV5ZUhkWGJGcFhZMGRvV2xaWFVrZGFWV1JQVWpKR1IyRkhiRk5pYTBwMlZtMTBVMU14VVhsVmEyUlZZbXR3YUZWdGVFdGpSbHB4VTIwNWJHSkhVbGxhVldNMVlWVXhXRlZzYUZkTlYyaDJWakJrUzFkV1ZuSlBWbHBvWVRGd1NWWkhlR0ZaVm1SR1RsWmFVRlp0YUZSVVZWcGFUVlphYzFwRVVtcE5WMUl3VlRKMGIyRkdTbk5UYkdoVlZteHdNMVl3V25KbFJtUnlaRWR3YVZacmNFbFdiR1EwWVRKR1YxTnVVbEJXUlRWWVZGYzFiMWRHYkhGVGExcHNVbTFTV2xkclZURlhSa3BaVVc1b1YxWXphSFpWVkVaYVpVWmtkVkpzVm1sV1IzaDZWMWQwWVdReVZrZFdibEpPVmxkU1YxUlhkSGRXTVZwMFkwZEdXR0pHY0ZoWk1HUnZWMjFGZVZWclpHRldWMUpRVlRGa1MxSXhjRWRhUms1WFYwVktNbFp0TVRCVk1VMTRWVmhzVm1FeVVsVlpiWFIzWVVaV2RFMVhPV3BTYkhCNFZrY3dOVll4V25OalJXaFlWa1UxZGxsV1ZYaGpiVXBGVld4a1RsWXlhREpXTVZwaFV6RktjMVJ1VWxCV2JGcFlXV3RvUTFkV1draGxSMFphVm0xU1IxUnNXbUZWUmxsNVlVaENWbUpIYUVOYVJFWmhZekZ3UlZWdGNFNVdNVWwzVmxSS01HRXhaRWhUYkdob1VqQmFWbFp0ZUhkTk1YQllaVWhLYkZZeFdrbGFSV1F3VlRKRmVsRllaRmhpUmxwb1dWUktSMWRHU2xsYVIzQlRWak5vV1ZkWGVHOVJNVkpIWTBab2FtVnJXbGhVVm1SVFpXeHNWbGRzVG1oV2EzQXhWVmMxYjFZeFdYcGhTRXBYVmtWYWVsWnFSbGRqTVdSellVZHNWMVp1UWpaV01XUXdXVmROZDAxSWFGaFhSM2hQVm14a1UxWXhVbGhrU0dSVFRWWktlbGxWYUd0WFIwcEhZMFpvV2sxSGFFeFdNbmhoVjBaV2NscEhSbGRXTVVwUlZsUkNhMUl4U1hsU2EyaHBVbXMxY0ZsVVFuZE5iRnAwVFZSQ1ZrMVZNVFJXVm1oelZsWmtTR1ZHV2xwV1JWb3pXVlZhVjJOV1RuUlBWbVJUWWtWd1dsWkhlR3BPVmxsNFYyNU9hbEpYYUZsV2ExVXhaR3hzVjFaWWFHcGlWWEJHVmxkNGExUnNXWGxoUkVwWFlXdEtjbFY2Umt0amF6VlhXa1phYVZKc2NGbFhWM2hoVW0xUmVGZHVVbXBTVjFKWFZGWmFkMDFHVm5Sa1J6bFdVbXh3TUZaWGN6VldNa1p5VjJ0NFZrMXVhR2haZWtaM1VsWldkR05GTlZkTlZXd3pWbXhTUzAxSFJYaGFSV2hVWWtkb2IxVnRNVzlaVmxweVZtMUdUazFXY0hsV01qRkhZV3hhY21ORVJsaGhNWEJRVmtkNFlXTnRTWHBhUm1ocFVteHdiMWRXVWt0U01WbDRWR3hzYWxKdVFrOVVWekZ2VjFaYVIxbDZSbWxOVjFKSVZqSTFSMVV5U2taalNFNVdZbFJHVkZZeWVHdGpiRnBWVW14b1UyRXpRbUZXVm1RMFl6RmtSMWR1VWxaaGJIQldWbTE0ZDJWc1duRlNiR1JxVFZkU2VsbFZaSE5oVmxweVkwWndWMkpIVGpSVWEyUlNaVlphY2xwR1pHbGlSWEJRVm0xNGExVXhaRWRWYkZwV1lUSlNjMVp0ZUV0bGJGcDBUVVJXVjAxRVJsZFpibkJMVm0xS1dWVnVXbGRoYTFwb1ZXMTRhMk50VmtkYVIyaG9UVEJLVWxac1kzaE9SbXhZVkZob2FsSlhhSEJWYlhNeFZERmFjMWRzY0d4aVJuQXdXbFZrUjFack1WWmlSRkphWVRGd2NsWXdaRXRqYlU1R1QxWmthVmRIWjNwV2FrSmhZekZrV0ZSclpHRlNiVkpVV1d0YWQwNVdXblJOVkVKYVZteEdORlp0ZUZkVWJFcElZM3ByUFE9PQ=='

if __name__ == '__main__':
    
    pass
    