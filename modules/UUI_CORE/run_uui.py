"""Run a .uui file with the restricted UUI 0.1 interpreter."""
import argparse
import json
from pathlib import Path
from uui_bridge import UUIBridge
from uui_lang import Runtime, UUIError

def main():
    p=argparse.ArgumentParser()
    p.add_argument('program',type=Path)
    p.add_argument('--database',default=':memory:')
    p.add_argument('--forge-pack',default=None)
    args=p.parse_args()
    if args.program.suffix!='.uui':p.error('Expected a .uui source file')
    bridge=UUIBridge(kira_db=args.database,forge_pack_path=args.forge_pack)
    try:
        result=Runtime(bridge).run(args.program.read_text(encoding='utf-8'))
        for row in result:print(json.dumps(row,ensure_ascii=False) if isinstance(row,dict) else row)
    except UUIError as e:p.exit(1,f'UUI error: {e}\n')
    finally:bridge.close()
if __name__=='__main__':main()
