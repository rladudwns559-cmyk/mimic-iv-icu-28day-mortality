"""리터럴을 동일 값의 상수로 옮긴 리팩터링만 AST로 증명하여 체크포인트를 유지한다.

일반적인 코드/기준 변경에는 사용할 수 없다. 기존 코드 백업과 현재 코드가
상수 치환 후 완전히 같은 AST일 때만 실행되며 입력 파일 지문도 다시 확인한다.
"""
import argparse
import ast
import copy
import hashlib
import json
from types import SimpleNamespace

from pipeline_common import CONFIG_PATH, ROOT, fingerprint, raw_path, save_json


def canonical(source):
    """최상단의 명명된 숫자 상수를 원래 리터럴로 되돌린 AST를 반환한다."""
    tree = ast.parse(source)
    names = {'NONNEGATIVE_MIN', 'SPO2_MIN_EXCLUSIVE', 'SPO2_MAX_INCLUSIVE', 'TEMPERATURE_MIN_C'}
    values = {}
    for statement in tree.body:
        if isinstance(statement,ast.Assign) and len(statement.targets)==1 and isinstance(statement.targets[0],ast.Name):
            name = statement.targets[0].id
            if name in names:
                ast.literal_eval(statement.value)  # 함수 호출 등은 허용하지 않는다.
                values[name] = statement.value
    tree.body = [s for s in tree.body if not (isinstance(s,ast.Assign) and
                 len(s.targets)==1 and isinstance(s.targets[0],ast.Name) and s.targets[0].id in values)]

    class Replace(ast.NodeTransformer):
        """상수명과 상수만으로 구성한 f-string만 정규화한다."""
        def visit_Name(self, node):
            return copy.deepcopy(values[node.id]) if node.id in values else node

        def visit_JoinedStr(self,node):
            node = self.generic_visit(node)
            chunks = []
            try:
                for v in node.values:
                    if isinstance(v,ast.Constant):
                        chunks.append(v.value)
                    elif isinstance(v,ast.FormattedValue) and v.conversion == -1 and v.format_spec is None:
                        chunks.append(str(ast.literal_eval(v.value)))
                    else:
                        return node
                return ast.Constant(value=''.join(chunks))
            except (ValueError,TypeError):
                return node

    return ast.dump(Replace().visit(tree),include_attributes=False)


def main():
    """동일 동작과 동일 입력을 증명한 뒤 지문 이력과 새 지문을 원자적으로 기록한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('old_code',type=str)
    parser.add_argument('--output-dir',type=str,required=True)
    args = parser.parse_args()
    from pathlib import Path
    old = Path(args.old_code).read_bytes()
    current = Path(__file__).with_name('pipeline_common.py').read_bytes()
    if canonical(old.decode('utf-8')) != canonical(current.decode('utf-8')):
        raise ValueError('동작 변경 감지: 상수 치환만으로 AST가 일치하지 않아 체크포인트 유지 불가')
    output = Path(args.output_dir)
    changes = []
    for kind in ['vital','lab']:
        mp = output/f'{kind}_audit/manifest.json'
        m = json.loads(mp.read_text(encoding='utf-8'))
        if not m['complete']:
            raise ValueError('추출 완료 체크포인트만 갱신할 수 있습니다.')
        a = SimpleNamespace(cohort=ROOT/'data/processed/cohort_base.csv', raw_dir=ROOT/'data/raw',
                            chunksize=m['signature']['chunksize'],max_chunks=m['signature']['max_chunks'])
        h = hashlib.sha256()
        for data in [a.cohort.read_bytes(),CONFIG_PATH.read_bytes(),old]:
            h.update(data)
        for name in ['d_items','d_labitems']:
            h.update(raw_path(a.raw_dir,name).read_bytes())
        new = fingerprint(a,raw_path(a.raw_dir,'chartevents' if kind=='vital' else 'labevents'),kind)
        expected = dict(new,version=h.hexdigest())
        if m['signature'] != expected:
            raise ValueError(f'{kind}: 기존 입력/코드 지문 불일치. 체크포인트를 갱신하지 않았습니다.')
        changes.append((mp,m,new))
    for mp,m,new in changes:
        m.setdefault('equivalent_refactors',[]).append(dict(old_signature=m['signature'],
            new_signature=new,proof='전체 모듈 AST가 명명된 숫자 상수 치환 후 동일'))
        m['signature'] = new
        save_json(mp,m)
    print('동일 AST·입력 검증 통과: 상수 리팩터링 체크포인트 갱신')


if __name__ == '__main__':
    main()
