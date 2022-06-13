import sys
import math 
import numpy as np

class FollowTheGap:
    BUBBLE_RADIUS = 30        # 위험 방울  (Minimize for Narrow asle)
    PREPROCESS_CONV_SIZE = 3  # 이동 평균 Window
    BEST_POINT_CONV_SIZE = 30 

    MAX_LIDAR_DIST = 30

    STRAIGHTS_SPEED = 0.1   # Modifying ...
    CORNERS_SPEED   = 0.1   # Modifying ...

    STRAIGHTS_STEERING_ANGLE = np.pi / 180   # 1 radian 
    
    def __init__(self):
        self.radians_per_elem = None

    def preprocess_lidar(self, ranges, angle_increment):

        # Lidar 분해각
        self.radians_per_elem = angle_increment        
        cut_to_90_reverse = np.flip(ranges[:90])
        cut_to_270_360 = ranges[270:]
        proc_ranges = np.append(cut_to_90_reverse, cut_to_270_360, axis = 0)
        
        proc_ranges = np.convolve(proc_ranges, np.ones(self.PREPROCESS_CONV_SIZE), 'same') / self.PREPROCESS_CONV_SIZE
        proc_ranges = np.clip(proc_ranges, 0, self.MAX_LIDAR_DIST)
        return proc_ranges


    def find_max_gap(self, free_space_ranges): # free_space_ranges 에 bubble zone 포함

        masked = np.ma.masked_where(free_space_ranges == 0, free_space_ranges)
        slices = np.ma.notmasked_contiguous(masked) 

        # loop문을 통해 maximum length free space 찾기
        max_len = slices[0].stop - slices[0].start
        chosen_slice = slices[0]
        for sl in slices[1:]:
            sl_len = sl.stop - sl.start
            if sl_len > max_len:
                max_len = sl_len
                chosen_slice = sl

        return chosen_slice.start, chosen_slice.stop # 시작과 끝점 return

    def find_best_point(self, start_i, end_i, ranges):
        averaged_max_gap = np.convolve(ranges[start_i:end_i], np.ones(self.BEST_POINT_CONV_SIZE),
                                       'same') / self.BEST_POINT_CONV_SIZE

        return averaged_max_gap.argmax() + start_i

    def get_angle(self, range_index): 

        LEFT_FLAG  = 0
        RIGHT_FLAG = 0
        
        if range_index <= 90:
            index     = 90 - range_index
            LEFT_FLAG = 1

        else:
            index      = range_index - 90
            RIGHT_FLAG = 1

        angle = index * self.radians_per_elem        

        return angle, LEFT_FLAG, RIGHT_FLAG

    def process_lidar(self, ranges, angle_increment):

        # Left Size, Right Side lidar datas        
        proc_ranges = self.preprocess_lidar(ranges, angle_increment)
        closest = proc_ranges.argmin() # Minimum distance index at left side

        # Eliminate all points inside 'bubble' (set them to zero)
        min_index = closest - self.BUBBLE_RADIUS
        max_index = closest + self.BUBBLE_RADIUS

        # 최대 index 최소 index 처리
        if min_index < 0:
            min_index = 0
        if max_index >= len(proc_ranges):
            max_index = len(proc_ranges) - 1

        # bubble 안에 있는 값들 모두 0으로 처리 ---- 0이 아닌 data들은 이제 free space 내의 데이터들이다.
        proc_ranges[min_index:max_index] = 0        

        # Find max length gap --- 가장 긴 Free Space의 시작 index와 끝 index
        gap_start, gap_end = self.find_max_gap(proc_ranges)

        # Find the best point in the gap ---- best index get !
        best = self.find_best_point(gap_start, gap_end, proc_ranges)

        # Publish Drive message --- modify 
        steering_angle, left, right = self.get_angle(best)

        if abs(steering_angle) > self.STRAIGHTS_STEERING_ANGLE: # STRAIGHTS_STEERING_ANGLE == 10도
            speed = self.CORNERS_SPEED
        else:
            speed = self.STRAIGHTS_SPEED

        return speed, steering_angle, best, left, right 
        
