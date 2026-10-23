"""Import newline-delimited JSON quantizer exports into UUI persistent SQLite."""
import argparse
import json
from uui_bridge import UUIBridge

def main():
    p=argparse.ArgumentParser()
    p.add_argument('jsonl',help='UTF-8 JSONL file of sensor observation envelopes')
    p.add_argument('--database',default='kira_uui.sqlite')
    args=p.parse_args()
    bridge=UUIBridge(kira_db=args.database)
    try:
        with open(args.jsonl,encoding='utf-8') as stream:
            for line_number,line in enumerate(stream,1):
                if not line.strip():continue
                try:
                    result=bridge.sensors.ingest(json.loads(line))
                except (ValueError,TypeError,json.JSONDecodeError) as exc:
                    p.exit(1,f'Line {line_number}: {exc}\n')
                print(json.dumps(result,ensure_ascii=False))
    finally:bridge.close()
if __name__=='__main__':main()
