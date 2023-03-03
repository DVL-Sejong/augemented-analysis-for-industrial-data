data = {}
js = {}
final ={}
final['name'] = 'a'
final['children'] = []
js['name'] = []
data['name'] = []
js['children'] = []
data['children'] = []
for x in range(len(a)):
    if a.iloc[x,:]['MAIN_OBJ'] != a.iloc[x-1,:]['MAIN_OBJ']:
        js['name'] = a.iloc[x,:]['RELATION']
        js['children'].append({"name": a.iloc[x,:]['SUB_OBJ']})
        data['name'] = a.iloc[x,:]['MAIN_OBJ']
        data['children'].append(js)
        final['children'].append(data)
        js = {}
        data = {}
        js['children'] = []
        data['children'] = []
    else:
