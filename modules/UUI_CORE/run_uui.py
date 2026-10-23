"""Run UUI source or compiled UUIX AST with the Python-hosted runtime."""
import argparse
import json
from pathlib import Path
from uui_bridge import UUIBridge
from uui_lang import Runtime, UUIError
from uui_compiler import compile_source,serialize,deserialize

def main():
    p=argparse.ArgumentParser()
    p.add_argument('program',type=Path)
    p.add_argument('--compile',action='store_true',help='Compile .uui into .uuix without running')
    p.add_argument('--database',default='kira_uui.sqlite',help='Persistent SQLite database (use :memory: for temporary runs)')
    p.add_argument('--forge-pack',default=None)
    args=p.parse_args()
    try:
        if args.compile:
            if args.program.suffix!='.uui':p.error('Compilation requires .uui')
            out=args.program.with_suffix('.uuix')
            out.write_bytes(serialize(compile_source(args.program.read_text(encoding='utf-8'))))
            print(out)
            return
        if args.program.suffix not in ('.uui','.uuix'):p.error('Expected .uui or .uuix')
        bridge=UUIBridge(kira_db=args.database,forge_pack_path=args.forge_pack)
        try:
            vm=Runtime(bridge)
            if args.program.suffix=='.uui':result=vm.run(args.program.read_text(encoding='utf-8'))
            else:result=vm.run_compiled(deserialize(args.program.read_bytes()))
            for row in result:print(json.dumps(row,ensure_ascii=False) if isinstance(row,dict) else row)
        finally:bridge.close()
    except (UUIError,OSError) as e:p.exit(1,f'UUI error: {e}\n')
if __name__=='__main__':main()
