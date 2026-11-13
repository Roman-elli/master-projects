import sys
import shlex
sys.modules['pipes'] = shlex

from mrjob.job import MRJob
from mrjob.step import MRStep

class MRMinTemperature(MRJob):

    def steps(self):
        return [
            MRStep(mapper=self.mapper_get_temps,
                   reducer=self.reducer_min_temp)
        ]

    def make_celsius(self, tenths_of_celsius):
        return float(tenths_of_celsius) / 10.0

    def mapper_get_temps(self, _, line):
        fields = line.split(',')
        station_id = fields[0]
        obs_type = fields[2]
        
        if obs_type == 'TMIN':
            temperature = self.make_celsius(fields[3])
            yield station_id, temperature

    def reducer_min_temp(self, station_id, temps):
        yield station_id, min(temps)

if __name__ == '__main__':
    MRMinTemperature.run()