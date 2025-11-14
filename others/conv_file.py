#!/usr/bin/env python
# coding: utf-8

# In[10]:


import nbformat

# IPYNB → PY 변환 (셀 정보 보존)
ipynb_file = '1_LM_Simple_Linear_to_Polynomial_Regression_df_w_answer.ipynb'  # 원본 파일명으로 변경

with open(ipynb_file, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

py_content = []
py_content.append('#!/usr/bin/env python')
py_content.append('# coding: utf-8')
py_content.append('')

for cell in nb.cells:
    if cell.cell_type == 'markdown':
        py_content.append('# %% [markdown]')
        for line in cell.source.split('\n'):
            py_content.append('# ' + line)
        py_content.append('')
    
    elif cell.cell_type == 'raw':
        py_content.append('# %% [raw]')
        for line in cell.source.split('\n'):
            py_content.append('# ' + line)
        py_content.append('')
    
    elif cell.cell_type == 'code':
        py_content.append('# %%')
        py_content.append(cell.source)
        py_content.append('')

# PY 파일 저장
py_file = ipynb_file.replace('.ipynb', '.py')
with open(py_file, 'w', encoding='utf-8') as f:
    f.write('\n'.join(py_content))

print(f"✅ IPYNB → PY 변환 완료: {py_file}")


# In[7]:


import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell, new_raw_cell

# PY → IPYNB 변환 (셀 정보 복원)
py_file = '2_LM_Simple_Linear_to_Polynomial_Regression_ndarray.py'  # 변환할 파일명

with open(py_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

cells = []
current_cell_lines = []
current_cell_type = None

for line in lines:
    line = line.rstrip('\n')
    
    if line.startswith('# %% [markdown]'):
        # 이전 셀 저장
        if current_cell_type and current_cell_lines:
            if current_cell_type == 'markdown':
                content = '\n'.join([l[2:] if l.startswith('# ') else l for l in current_cell_lines])
                cells.append(new_markdown_cell(content))
            elif current_cell_type == 'raw':
                content = '\n'.join([l[2:] if l.startswith('# ') else l for l in current_cell_lines])
                cells.append(new_raw_cell(content))
            elif current_cell_type == 'code':
                content = '\n'.join(current_cell_lines)
                cells.append(new_code_cell(content))
        
        current_cell_type = 'markdown'
        current_cell_lines = []
    
    elif line.startswith('# %% [raw]'):
        # 이전 셀 저장
        if current_cell_type and current_cell_lines:
            if current_cell_type == 'markdown':
                content = '\n'.join([l[2:] if l.startswith('# ') else l for l in current_cell_lines])
                cells.append(new_markdown_cell(content))
            elif current_cell_type == 'raw':
                content = '\n'.join([l[2:] if l.startswith('# ') else l for l in current_cell_lines])
                cells.append(new_raw_cell(content))
            elif current_cell_type == 'code':
                content = '\n'.join(current_cell_lines)
                cells.append(new_code_cell(content))
        
        current_cell_type = 'raw'
        current_cell_lines = []
    
    elif line.startswith('# %%'):
        # 이전 셀 저장
        if current_cell_type and current_cell_lines:
            if current_cell_type == 'markdown':
                content = '\n'.join([l[2:] if l.startswith('# ') else l for l in current_cell_lines])
                cells.append(new_markdown_cell(content))
            elif current_cell_type == 'raw':
                content = '\n'.join([l[2:] if l.startswith('# ') else l for l in current_cell_lines])
                cells.append(new_raw_cell(content))
            elif current_cell_type == 'code':
                content = '\n'.join(current_cell_lines)
                cells.append(new_code_cell(content))
        
        current_cell_type = 'code'
        current_cell_lines = []
    
    elif line.startswith('#!/usr/bin/env python') or line.startswith('# coding:'):
        # 헤더 라인은 무시
        continue
    
    else:
        # 현재 셀에 라인 추가
        if current_cell_type:
            current_cell_lines.append(line)

# 마지막 셀 저장
if current_cell_type and current_cell_lines:
    if current_cell_type == 'markdown':
        content = '\n'.join([l[2:] if l.startswith('# ') else l for l in current_cell_lines])
        cells.append(new_markdown_cell(content))
    elif current_cell_type == 'raw':
        content = '\n'.join([l[2:] if l.startswith('# ') else l for l in current_cell_lines])
        cells.append(new_raw_cell(content))
    elif current_cell_type == 'code':
        content = '\n'.join(current_cell_lines)
        cells.append(new_code_cell(content))

# 노트북 생성 및 저장
nb = new_notebook()
nb['cells'] = cells
ipynb_file = py_file.replace('.py', '_restored.ipynb')

with open(ipynb_file, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print(f"✅ PY → IPYNB 변환 완료: {ipynb_file}")
print(f"📊 총 {len(cells)}개 셀 복원")


# In[ ]:





# In[ ]:




