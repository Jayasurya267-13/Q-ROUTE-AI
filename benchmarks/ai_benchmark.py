"""Measure actual local CPU inference latency."""
from pathlib import Path
import sys,time,statistics
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ai.inference import TrafficPredictor

def main(calls=100):
    predictor=TrafficPredictor(); sample=dict(traffic_volume=2500,temp=288.5,rain_1h=0,snow_1h=0,clouds_all=40,hour=8,day_of_week=2,month=10)
    predictor.predict(**sample)
    times=[]
    for _ in range(calls):
        t=time.perf_counter();predictor.predict(**sample);times.append((time.perf_counter()-t)*1000)
    print(f"Local CPU prototype benchmark ({calls} calls)")
    print(f"Average: {statistics.mean(times):.3f} ms | min: {min(times):.3f} ms | max: {max(times):.3f} ms")
if __name__=="__main__":main()
