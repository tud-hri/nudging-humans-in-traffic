import glob
import os
import sys

from exp_info_ui import ExpInfoUI

try:
    sys.path.append(glob.glob('C:\carla-v0.9.9\PythonAPI\carla\dist\carla-0.9.9-py3.8-win-amd64.egg')[0])
    import carla
except BaseException:
    pass

import pygame
import math
import time
import random
import numpy as np
import tkinter as tkr
import csv
import numpy.matlib


class LTAPCarlaClient():
    block_size = 150
    lane_width = 3.5

    bot_colors = []

    #    bot_blueprint_names = ['vehicle.tesla.model3', 'vehicle.nissan.micra', 'vehicle.volkswagen.t2',
    #                           'vehicle.mini.cooperst', 'vehicle.citroen.c3']
    bot_blueprint_names = ['vehicle.tesla.model3']

    bot_blueprint_colors = ['147,130,127', '107,162,146', '53,206,141', '188,216,183', '224,210,195',
                            '255,237,101', '180,173,234']

    def __init__(self):
        try:
            self.exp_info = self.get_exp_info()
            self.n_routes_per_session = 5

            # self.set_ff_gain()
            self.initialize_log()
            self.client = carla.Client('localhost', 2000)
            self.client.set_timeout(2.0)
            pygame.init()
            self.world = self.client.get_world()

            self.world.set_weather(carla.WeatherParameters.ClearSunset)
            # Set Weather - Only affects visuals
            # Setting the weather has no effect on the physics

            # self.tta_conditions = [4, 5, 6]
            # self.tta_conditions = [4.5, 5.5]
            self.tta_conditions = [4, 5]
            # self.bot_distance_values = [90, 120, 150]
            self.bot_distance_values = [80]
            self.accl_conditions = [[0.0, 0.0, 0.0, 0.0],
                                    [0.0, -4, -4, 0.0],     # Pure Deceleration
                                    [0.0, -4, 4, 0.0],      # Flip Deceleration
                                    [0.0, 4, -4, 0.0],      # Flip Acceleration
                                    [0.0, 4, 4, 0.0],       # Pure Acceleration
                                    [0.0, 0.0, 0.0, 0.0],  # For very short tta condition - distance short
                                    [0.0, 0.0, 0.0, 0.0]]  # For very long tta condition - distance Long
            # self.accl_conditions = [[0.0, 0.0, 0.0, 0.0],
            #                         [0.0, -3, 3, 0.0],
            #                         [0.0, 3, -3, 0.0],
            #                         [0.0, -5, 5, 0.0],
            #                         [0.0, 5, -5, 0.0],
            #                         [0.0, 0.0, 0.0, 0.0],  # For very short tta condition - distance short
            #                         [0.0, 0.0, 0.0, 0.0]]  # For very long tta condition - distance Long


            self.accl_time_conditions = np.matlib.repmat([0., 0.25, 1.25, 2.25], (5 + 2), 1)  # 5 accl profiles + 2 dummy
            # For now the acc_time are the same for all the acceleration profiles.

            # x and y indices of the intersection
            # (0, 0) is the intersection in the bottom left corner of the map
            # (4, 4) is the intersection in the top right corner of the map
            self.origin = np.array([0.0, 0.0])

            # (1,0) is east, (-1,0) is west, (0,1) is north, (0,-1) is south
            self.active_intersection = np.array([1.0, 0.0])

            self.sound_cues = {(1, 1): 'next_turn_left',
                               (1, 0): 'next_go_straight',
                               (1, -1): 'next_turn_right',
                               (2, 1): 'turn_left',
                               (2, 0): 'go_straight',
                               (2, -1): 'turn_right'}

            self.ego_actor = None
            self.bot_actor = None
            self.bot_actor_blueprints = [random.choice(self.world.get_blueprint_library().filter(bp_name))
                                         for bp_name in self.bot_blueprint_names]

            self.empty_control = carla.VehicleControl(hand_brake=False, reverse=False, manual_gear_shift=False)
            self.control = self.empty_control
            self.let_pass = 0
            self.subjective_good = 0
            self.subjective_bad = 0

        except KeyboardInterrupt:
            for actor in self.world.get_actors():
                actor.destroy()

    def set_ff_gain(self, gain=35):
        ffset_cmd = 'ffset /dev/input/event%i -a %i'
        for i in range(5, 10):
            os.system(ffset_cmd % (i, gain))

    def generate_tta_values(self):
        tta_values = []
        for tta in self.tta_conditions:
            # 5 is the number of left turns per route per tta
            tta_values = np.append(tta_values, np.ones(12) * tta)
        AccIndex = range(5)  # 5 accl coniditons
        AccIndexRep = np.matlib.repmat(AccIndex, 1, 2)
        AccIndexRep = np.append(AccIndexRep, [5, 6])  # Adding the indices for the Dummy Trials - 2 per tta
        AccIndexRep = np.matlib.repmat(AccIndexRep, 1, 2)  # 4*5 Trials + 2 Dummy trials
        AccIndexRep = np.squeeze(AccIndexRep)

        ii = np.arange(len(tta_values))

        np.random.shuffle(ii)
        print('ii = ', ii)

        # np.random.shuffle(tta_values)

        tta_values = tta_values[ii]

        AccIndexRep = AccIndexRep[ii]
        print(tta_values)
        print(AccIndexRep)

        for ii in range(len(tta_values)):
            print(tta_values[ii], AccIndexRep[ii])

        return tta_values, AccIndexRep

    def initialize_log(self):
        log_directory = 'data'
        self.log_file_path = os.path.join(log_directory, str(self.exp_info['subj_id']) + '_' +
                                          str(self.exp_info['session']) + '_' +
                                          self.exp_info['start_time'] + '.txt')
        with open(self.log_file_path, 'w') as fp:
            writer = csv.writer(fp, delimiter='\t')
            writer.writerow(['subj_id', 'session', 'route', 'intersection_no',
                             'intersection_x', 'intersection_y', 'turn_direction', 't',
                             'ego_distance_to_intersection', 'tta_condition', 'd_condition', 'v_condition', 'truck_angle', 'bot_angle',
                             'accl_profile_values', 'accl_profile_times',
                             'ego_x', 'ego_y', 'ego_vx', 'ego_vy', 'ego_ax', 'ego_ay', 'ego_yaw',
                             'bot_x', 'bot_y', 'bot_vx', 'bot_vy', 'bot_ax', 'bot_ay', 'bot_yaw',
                             'throttle', 'brake', 'steer', 'let_pass', 'subjective_good', 'subjective_bad',
                             'truck_x', 'truck_y', 'truck_vx', 'truck_vy', 'truck_ax', 'truck_ay', 'truck_yaw'])

    def write_log(self, log):
        with open(self.log_file_path, 'a') as fp:
            writer = csv.writer(fp, delimiter='\t', )
            writer.writerows(log)

    def get_exp_info(self):
        root = tkr.Tk()
        app = ExpInfoUI(master=root)
        app.mainloop()
        exp_info = app.exp_info
        root.destroy()

        return exp_info

    def rotate(self, vector, angle):
        c, s = np.cos(angle), np.sin(angle)
        return np.squeeze(np.asarray(np.dot(np.matrix([[c, -s], [s, c]]), vector)))

    def update_ego_control(self):
        reverse = False
        K1 = 1.0  # 0.55
        steerCmd = K1 * math.tan(1.1 * self.joystick.get_axis(0))

        K2 = 1.6  # 1.6
        # throttleCmd = K2 + (2.05 * math.log10(
        # -0.7 * self.joystick.get_axis(1) + 1.4) - 1.2) / 0.92
        throttleCmd = K2 + (2.05 * math.log10(
            -0.9 * self.joystick.get_axis(1) + 1.4) - 1.2) / 0.92
        if throttleCmd <= 0:
            throttleCmd = 0
        elif throttleCmd > 1:
            throttleCmd = 1

        # cap the speed at 20 m/s
        speed = np.sqrt(self.ego_actor.get_velocity().x ** 2 + self.ego_actor.get_velocity().y ** 2)
        if speed > 20:
            throttleCmd = 0

        # brakeCmd = 1.6 + (2.05 * math.log10(
        #     -0.7 * self.joystick.get_axis(2) + 1.4) - 1.2) / 0.92
        brakeCmd = 1.6 + (2.05 * math.log10(
            -1 * self.joystick.get_axis(2) + 1.4) - 1.2) / 0.92
        if brakeCmd <= 0:
            brakeCmd = 0
        elif brakeCmd > 1:
            brakeCmd = 1

        if self.joystick.get_button(5):
            reverse = True
        else:
            # elif self.joystick.get_button(4):
            reverse = False

        for event in pygame.event.get():
            if event.type == pygame.KEYUP:
                if event.key == pygame.locals.K_ESCAPE:
                    raise KeyboardInterrupt

        button_list = []

        # # For checking the indices associated with each button on the Joystick
        # # Uncomment the following code block and press individual buttons when you run the code.
        # for bb in range(self.joystick.get_numbuttons()):
        #     button_state = self.joystick.get_button(bb)
        #     button_list.append(bb)
        #     button_list.append(button_state)
        # print('Buttons : ',button_list)
        # # ########################################################################################

        self.control.throttle = throttleCmd
        self.control.steer = steerCmd
        self.control.brake = brakeCmd
        self.control.reverse = reverse


        self.let_pass_last = self.let_pass
        self.subjective_bad_last = self.subjective_bad

        self.let_pass = self.joystick.get_button(4)  # The right paddle

        if (self.let_pass == 1) & (self.let_pass_last == 0) :
            sound = pygame.mixer.Sound('sounds/LetPass.wav')
            sound.set_volume(0.1)
            sound.play()

        self.subjective_good = self.joystick.get_button(3)  # The triangle button
        self.subjective_bad = self.joystick.get_button(0)  # The X button

        if (self.subjective_bad == 1) & (self.subjective_bad_last == 0):
            sound = pygame.mixer.Sound('sounds/SubjectiveBad.wav')
            sound.set_volume(0.2)
            sound.play()



        self.ego_actor.apply_control(self.control)

    def spawn_ego_car(self):
        '''
        To shift the starting position from the center of the intersection to the lane where
        the driver can start driving towards the first active intersection, we rotate the
        heading direction 90` clockwise (-np.pi/2), and shift the origin towards that direction by half lane width
        '''
        start_position = self.origin * self.block_size + \
                         self.rotate(self.active_intersection - self.origin, -np.pi / 2) * self.lane_width / 2
        self.ego_start_position = self.world.get_map().get_waypoint(
            carla.Location(x=start_position[0], y=-start_position[1], z=0))

        # vehicle_bp_library = self.world.get_blueprint_library().filter('vehicle.*')
        # for items in vehicle_bp_library:
        #     print(items.id[8:])

        # ego_bp = random.choice(self.world.get_blueprint_library().filter('vehicle.tesla.model3'))
        ego_bp = random.choice(self.world.get_blueprint_library().filter('vehicle.hapticslab.audi'))

        self.ego_actor = self.world.spawn_actor(ego_bp, self.ego_start_position.transform)
        self.ego_actor.set_autopilot(False)

    def spawn_bot(self, distance_to_intersection, speed):
        bot_bp = random.choice(self.bot_actor_blueprints)
        bot_bp.set_attribute('color', random.choice(self.bot_blueprint_colors))

        ego_direction = self.active_intersection - self.origin

        spawn_location = self.active_intersection * self.block_size + \
                         distance_to_intersection * (ego_direction) + \
                         (self.lane_width / 2) * np.around(self.rotate(ego_direction, np.pi / 2))
        spawn_waypoint = self.world.get_map().get_waypoint(
            carla.Location(x=spawn_location[0], y=-spawn_location[1], z=0))

        self.bot_spawn_location_target = carla.Location(x=spawn_location[0], y=-spawn_location[1], z=0)

        self.bot_actor = self.world.spawn_actor(bot_bp, spawn_waypoint.transform)

        self.bot_velocity = speed * np.around(self.rotate(ego_direction, np.pi))  # .astype(int)

        self.bot_direction = np.around(self.rotate(ego_direction, np.pi))  # The direction of bot velocity
        self.last_bot_spawn_time = time.time()

        self.bot_actor.set_velocity(carla.Vector3D(self.bot_velocity[0], -self.bot_velocity[1], 0))

    def spawn_truck(self, distance_to_intersection, speed):
        truck_bp = random.choice(self.world.get_blueprint_library().filter('vehicle.carlamotors.carlacola'))

        ego_direction = self.active_intersection - self.origin

        spawn_location = self.active_intersection * self.block_size + \
                         (- distance_to_intersection) * (ego_direction) + \
                         (- self.lane_width / 2) * np.around(self.rotate(ego_direction, np.pi / 2))
        spawn_waypoint = self.world.get_map().get_waypoint(
            carla.Location(x=spawn_location[0], y=-spawn_location[1], z=0))

        self.truck_actor = self.world.spawn_actor(truck_bp, spawn_waypoint.transform)

        self.truck_velocity = speed * np.around(self.rotate(ego_direction, 0))  # .astype(int)

        self.truck_direction = np.around(self.rotate(ego_direction, 0))  # The direction of truck velocity

        self.truck_actor.set_velocity(carla.Vector3D(self.truck_velocity[0], -self.truck_velocity[1], 0))

        print('Truck Spawned !', 'Ego Direction :', ego_direction)

    def update_truck_control(self, truck_max_speed, truck_accl):
        # if not self.bot_actor is None: # When the truck motion was tied to spawing of the bot
        if self.truck_move is True:
            truck_speeding_time = max(0, time.time() - self.truck_move_start_time)
            truck_speed = np.minimum(truck_max_speed, truck_accl * (truck_speeding_time)**2)
        else:
            truck_speed = 0
        # print('Truxk Speed Update: ',truck_speed,' Bot Direction: ',self.truck_direction)
        self.truck_velocity = truck_speed * self.truck_direction
        # print(self.truck_velocity)
        self.truck_actor.set_velocity(carla.Vector3D(self.truck_velocity[0], -self.truck_velocity[1], 0))

    # Older update_bot_control for constant velocity
    #    def update_bot_control(self, max_speed):
    #        self.bot_actor.set_velocity(carla.Vector3D(self.bot_velocity[0], -self.bot_velocity[1], 0))

    def update_bot_control(self, V0, acc, acc_t):
        if self.bot_move is True:

            # acc - accelaration combination
            # acc_t = Decision points in seconds from start
            # t = time.time() - self.last_bot_spawn_time
            t = time.time() - self.bot_move_start_time

            if ((t > acc_t[0]) & (t < acc_t[1])):
                speed_dynamic = V0 + acc[0] * (t - acc_t[0])
            elif ((t > acc_t[1]) & (t < acc_t[2])):
                speed_dynamic = V0 + acc[0] * (acc_t[1] - acc_t[0]) + acc[1] * (t - acc_t[1])
            elif ((t > acc_t[2]) & (t < acc_t[3])):
                speed_dynamic = V0 + acc[0] * (acc_t[1] - acc_t[0]) + acc[1] * (acc_t[2] - acc_t[1]) + acc[2] * (t - acc_t[2])
            elif (t > acc_t[3]):
                speed_dynamic = V0 + acc[0] * (acc_t[1] - acc_t[0]) + acc[1] * (acc_t[2] - acc_t[1]) + acc[2] * (acc_t[3] - acc_t[2]) + acc[3] * (t - acc_t[3])
        else:
            speed_dynamic = 0

        bot_velocity_dynamic = speed_dynamic * self.bot_direction
        self.bot_actor.set_velocity(carla.Vector3D(bot_velocity_dynamic[0], -bot_velocity_dynamic[1], 0))

    def calculate_bot_start_speed(self, distance, tta, acc, acc_t):

        # TIme windows
        dt1 = acc_t[1] - acc_t[0]
        dt2 = acc_t[2] - acc_t[1]
        dt3 = acc_t[3] - acc_t[2]
        dt4 = tta - acc_t[3]  # t[4] = tta

        # Terms in the equation to solve for starting speed V0
        x1 = acc[0] * dt1 ** 2 / 2
        x2 = (2 * acc[0] * dt1 + acc[1] * dt2) * dt2 / 2
        x3 = (2 * acc[0] * dt1 + 2 * acc[1] * dt2 + acc[2] * dt3) * dt3 / 2
        x4 = (2 * acc[0] * dt1 + 2 * acc[1] * dt2 + 2 * acc[2] * dt3 + acc[3] * dt4) * dt4 / 2

        # Initial Bot_speed
        bot_start_speed = (distance - (x1 + x2 + x3 + x4)) / (tta - acc_t[0])

        return bot_start_speed

    def get_anlge_of_sight(self):
        # Just checking if the function calls work
        # if not self.ego_actor is None:
        # print('Ego Actor Yaw : ',self.ego_actor.get_transform().rotation.yaw)
        # print(ego_location)
        # print('Ego: ', self.ego_actor.get_tra)
        #     print('Bounding Box Ego: ',self.ego_actor.bounding_box)
        #     print('Bounding Box Ego - Location x,y,z: ',self.ego_actor.bounding_box.location.x,self.ego_actor.bounding_box.location.y,self.ego_actor.bounding_box.location.z,' - Extent x,y,z: ',self.ego_actor.bounding_box.extent.x,self.ego_actor.bounding_box.extent.y,self.ego_actor.bounding_box.extent.z )
        # if not self.bot_actor is None:
        # print('Bounding Box Bot: ', self.bot_actor.bounding_box)
        # if not self.truck_actor is None:
        # print('Bounding Box Truck: ', self.truck_actor.bounding_box)

        # The real function content
        # A - Truck
        # B - Ego
        # C - Bot

        camera_offset_ego_bodyframe = np.array([.10, -.40])  # [10,-40,80] from the blueprint in UE
        ego_location = self.ego_actor.get_location()
        ego_yaw = self.ego_actor.get_transform().rotation.yaw
        camera_offset_ego_worldframe = self.rotate(camera_offset_ego_bodyframe, np.radians(ego_yaw))

        B = np.array([ego_location.x, ego_location.y]) + camera_offset_ego_worldframe
        B_carla_location = carla.Location(B[0], B[1], 0)
        # print('B (ego) :',B)

        if not self.bot_actor is None:
            bot_location = self.bot_actor.get_location()
            bot_yaw = self.bot_actor.get_transform().rotation.yaw
            bot_corner_bodyframe = np.array([self.bot_actor.bounding_box.location.x, self.bot_actor.bounding_box.location.y]) + \
                                   np.array([self.bot_actor.bounding_box.extent.x, self.bot_actor.bounding_box.extent.y])
            # Center of Bounding Box
            # [+X,+Y] for NE corner (in BodyFrame)
            bot_corner_worldframe = self.rotate(bot_corner_bodyframe, np.radians(bot_yaw))
            C = np.array([bot_location.x, bot_location.y]) + bot_corner_worldframe
            # print('C (bot) :', C)
        else:
            C = B
        C_carla_location = carla.Location(C[0], C[1], 0)

        if not self.truck_actor is None:
            truck_location = self.truck_actor.get_location()
            truck_yaw = self.truck_actor.get_transform().rotation.yaw
            truck_corner_bodyframe = np.array([self.truck_actor.bounding_box.location.x, self.truck_actor.bounding_box.location.y]) + \
                                     np.array([-self.truck_actor.bounding_box.extent.x, -self.truck_actor.bounding_box.extent.y])
            # Center of Bounding Box
            # [-X,-Y] for SW corner (in BodyFrame)
            truck_corner_worldframe = self.rotate(truck_corner_bodyframe, np.radians(truck_yaw))
            A = np.array([truck_location.x, truck_location.y]) + truck_corner_worldframe
            # print('A (truck) :', A)
        else:
            A = B
        A_carla_location = carla.Location(A[0], A[1], 0)

        BA = A - B
        BC = C - B

        self.truck_angle = np.arctan2(BA[1], BA[0])
        self.bot_angle = np.arctan2(BC[1], BC[0])

        angle = self.truck_angle - self.bot_angle
        angle = np.degrees(angle)

        # To draw arrows to active intersection if 'enter' button on the Joystick is pressed
        if self.joystick.get_button(23):
            self.world.debug.draw_arrow(B_carla_location, self.origin_loc, persistent_lines=False, life_time=0.01, color=carla.Color(0, 255, 0, 0))
            self.world.debug.draw_arrow(self.origin_loc, self.active_intersection_loc, persistent_lines=False, life_time=0.01)

        if angle > 0.0:
            bot_arrow_color = carla.Color(0, 255, 0, 0)
        else:
            bot_arrow_color = carla.Color(255, 0, 0, 0)

        if (not self.truck_actor is None) & (self.debug_flag is True):
            # self.world.debug.draw_arrow(ego_location, truck_location,persistent_lines=False,life_time=0.01)
            self.world.debug.draw_arrow(B_carla_location, A_carla_location, persistent_lines=False, life_time=0.01)
        if (not self.bot_actor is None) & (self.debug_flag is True):
            # self.world.debug.draw_arrow(ego_location,bot_location,persistent_lines=False,life_time=0.01,color=bot_arrow_color)
            self.world.debug.draw_arrow(B_carla_location, C_carla_location, persistent_lines=False, life_time=0.01, color=bot_arrow_color)
            # print('C (bot) :', C)

        return angle

    def play_sound_cue(self, number, direction):
        sound_filename = '%s.wav' % (self.sound_cues[(number, direction)])
        file_path = os.path.join('sounds', sound_filename)
        sound = pygame.mixer.Sound(file_path)
        sound.set_volume(0.5)
        sound.play()

    def initialize_noise_sound(self):
        file_path = 'sounds/tesla_noise.wav'
        self.noise_sound = pygame.mixer.Sound(file_path)
        self.noise_sound.set_volume(0.1)
        self.noise_sound.play(loops=-1)

    def get_actor_state(self, actor):
        state = ([actor.get_transform().location.x, -actor.get_transform().location.y,
                  actor.get_velocity().x, -actor.get_velocity().y,
                  actor.get_acceleration().x, -actor.get_acceleration().y,
                  actor.get_transform().rotation.yaw]
                 if not (actor is None) else np.zeros(7).tolist())
        return list(['%.4f' % value for value in state])

    def update_log(self, log, values_to_log):
        log.append((values_to_log + \
                    self.get_actor_state(self.ego_actor) + \
                    self.get_actor_state(self.bot_actor) + \
                    list(['%.4f' % value for value in [self.control.throttle, self.control.brake, self.control.steer]]) + \
                    list(['%i' % value for value in [self.let_pass, self.subjective_good, self.subjective_bad]]) + \
                    self.get_actor_state(self.truck_actor)))

    def run(self):
        try:
            print(self.exp_info)
            first_route = self.exp_info['route']
            self.debug_flag = False  # Set to true to draw debug arrows

            for i in range(first_route, self.n_routes_per_session + 1):
                # tta_values = self.generate_tta_values()
                tta_values, accl_indices = self.generate_tta_values()
                self.origin = np.array([0.0, 0.0])
                self.active_intersection = np.array([1.0, 0.0])

                self.joystick = pygame.joystick.Joystick(0)
                self.joystick.init()

                self.spawn_ego_car()

                self.initialize_noise_sound()
                # in the first session, we go through routes 1 to 4, in the second session, routes 5 to 8
                route_number = i + (self.exp_info['session'] - 1) * self.n_routes_per_session
                # in the path input file, -1 is turn right, 1 is turn left, 0 is go straight
                route = np.loadtxt(os.path.join('routes', 'route_%i.txt' % (route_number)))
                tta_condition = tta_values[-1]
                for j, current_turn in enumerate(route):
                    # if the current turn is left, set TTA for this trial
                    # and drop the current TTA value from the list
                    # the same TTA will be used for next trials until there's another left turnx
                    if (current_turn == 1):
                        tta_condition = tta_values[-1]
                        tta_values = tta_values[:-1]
                        accl_index = accl_indices[-1]
                        accl_indices = accl_indices[:-1]

                    # distance to the center of the ego car
                    d_condition = random.choice(self.bot_distance_values)

                    if (accl_index == 5):  # Dummy short
                        d_condition = 60
                        tta_condition = 1;

                    if (accl_index == 6):  # Dummy long
                        d_condition = 180
                        tta_condition = 10;

                    # bot_accelerations = [0., -5., 5., 0.]
                    # bot_acceleration_times = [0., 0.5, 1.5, 2.5]

                    bot_accelerations = self.accl_conditions[accl_index]
                    bot_acceleration_times = self.accl_time_conditions[accl_index]

                    bot_speed = d_condition / tta_condition  # Old way of finding constant bot_speed
                    # bot_speed = self.calculate_bot_start_speed( d_condition, tta, bot_accelerations, bot_acceleration_times)

                    is_turn_completed = False
                    is_at_active_intersection = False
                    is_first_cue_played = False
                    is_second_cue_played = False
                    self.truck_move = False
                    self.bot_move = False
                    self.truck_angle = 0.0
                    self.bot_angle = 0.0

                    truck_speed = 60
                    truck_accl = 60
                    # ruck_accl = 60
                    truck_distance_from_intersection = 0
                    bot_distance_from_spawn_location = 99999
                    self.truck_actor = None

                    intersection_coordinates = (self.active_intersection[0] * self.block_size,
                                                self.active_intersection[1] * self.block_size)

                    # whenever we exchange y-coordinates with Carla server, we invert the sign
                    self.origin_loc = carla.Location(x=(self.origin[0]*self.block_size),
                                                                  y=-(self.origin[1]*self.block_size),
                                                                  z=0.0)

                    # whenever we exchange y-coordinates with Carla server, we invert the sign
                    self.active_intersection_loc = carla.Location(x=intersection_coordinates[0],
                                                                  y=-intersection_coordinates[1],
                                                                  z=0.0)
                    trial_log = []
                    trial_start_time = time.time()

                    print('Current trial: %i, turn %f, TTA %f, bot speed %f, distance %f' %
                          (j + 1, current_turn, tta_condition, bot_speed, d_condition))
                    print('Acceleration Profile :', bot_accelerations)

                    while not is_turn_completed:
                        t = time.time() - trial_start_time
                        speed = np.sqrt(self.ego_actor.get_velocity().x ** 2 + self.ego_actor.get_velocity().y ** 2)
                        ego_distance_to_intersection = self.ego_actor.get_location().distance(self.active_intersection_loc)
                        '''
                        'subj_id', 'session', 'route', 'intersection_no',
                        'intersection_x', 'intersection_y', 'turn_direction', 't',
                        'ego_distance_to_intersection', 'tta_condition', 'd_condition', 'v_condition , 'truck_angle', 'bot_angle', 'accl_values', 'accl_times'
                        '''
                        values_to_log = list(['%i' % value for value in
                                              [self.exp_info['subj_id'], self.exp_info['session'], route_number, j + 1,
                                               intersection_coordinates[0], intersection_coordinates[1], current_turn]]) \
                                        + list(['%.4f' % value for value in
                                                [t, ego_distance_to_intersection, tta_condition, d_condition, bot_speed, self.truck_angle, self.bot_angle]]) \
                                        + list(['%s' % value for value in
                                                [bot_accelerations, bot_acceleration_times]])

                        self.update_log(trial_log, values_to_log)

                        self.update_ego_control()

                        if not self.bot_actor is None:
                            # self.update_bot_control(bot_speed) # Old Command
                            self.update_bot_control(bot_speed, bot_accelerations, bot_acceleration_times)
                            bot_distance_from_spawn_location = self.bot_actor.get_location().distance(self.bot_spawn_location_target)
                            if self.debug_flag is True:
                                print('Angle of Sight :', angle_of_sight, ' Bot Location:', self.bot_actor.get_location(), ' Delta =', bot_distance_from_spawn_location,
                                      self.bot_move)

                        if not self.truck_actor is None:
                            self.update_truck_control(truck_speed, truck_accl)

                        self.noise_sound.set_volume(0.05 + speed / 20)

                        angle_of_sight = self.get_anlge_of_sight()  # Placed here just for testing the angle of sight function
                        if self.debug_flag is True:
                            print('Angle of Sight :', angle_of_sight)

                        if ((not is_first_cue_played) & (ego_distance_to_intersection < (4 / 5) * self.block_size)):
                            self.play_sound_cue(1, current_turn)
                            is_first_cue_played = True
                            if (self.truck_actor is None) & ((current_turn == 1) or (current_turn == -1)):
                                self.spawn_truck(truck_distance_from_intersection, truck_speed)
                        # When the driver approaches the intersection, we play the second sound cue and destroy the bot at the previous intersection
                        elif ((not is_second_cue_played) & (ego_distance_to_intersection < (1 / 5) * self.block_size)):
                            self.play_sound_cue(2, current_turn)
                            is_second_cue_played = True
                        elif ((not is_at_active_intersection) & (ego_distance_to_intersection < 15)):
                            is_at_active_intersection = True
                        # if at the left turn, wait until almost a full stop before spawning a bot
                        elif ((current_turn == 1) & (is_at_active_intersection) & (speed < 1) &
                              (self.bot_actor is None)):
                            self.truck_move = True
                            self.truck_move_start_time = time.time()
                            self.spawn_bot(distance_to_intersection=d_condition - ego_distance_to_intersection,
                                           speed=0)
                        # if at the right turn, don't wait for slowdown when spawning a bot
                        elif ((current_turn == -1) & (is_at_active_intersection) & (self.bot_actor is None)):
                            self.truck_move = True
                            self.truck_move_start_time = time.time()
                            self.spawn_bot(75, 0)
                            self.bot_move = True
                            self.bot_move_start_time = time.time()
                            # truck_speed = 30.0
                        elif ((angle_of_sight > 0.0) & (self.bot_move is False) & (not self.bot_actor is None) & (bot_distance_from_spawn_location < 0.01)):
                            self.bot_move = True
                            self.bot_move_start_time = time.time()
                        # When the driver leaves the intersection we designate the next intersection as active and destroy the bot
                        elif ((is_at_active_intersection) & (ego_distance_to_intersection > 15)):
                            print('updating origin and active intersection')
                            current_direction = self.active_intersection - self.origin
                            print(current_direction)
                            new_origin = self.active_intersection
                            print(new_origin)
                            new_active_intersection = (self.active_intersection +
                                                       self.rotate(current_direction, np.pi / 2 * current_turn))
                            print(new_active_intersection)
                            self.origin = new_origin
                            self.active_intersection = new_active_intersection

                            if (not (self.bot_actor is None)):
                                self.bot_actor.destroy()
                                self.bot_actor = None

                            if (not (self.truck_actor is None)):
                                self.truck_actor.destroy()
                                self.truck_actor = None
                                truck_speed = 0

                            self.truck_move = False
                            self.bot_move = False
                            bot_distance_from_spawn_location = 9999

                            is_turn_completed = True

                        time.sleep(0.01)

                    self.write_log(trial_log)

                self.noise_sound.stop()

                if (not (self.ego_actor is None)):
                    self.ego_actor.destroy()
                    self.ego_actor = None
                if (not (self.bot_actor is None)):
                    self.bot_actor.destroy()
                    self.bot_actor = None
                self.joystick.quit()

                time.sleep(5.0)

        except KeyboardInterrupt:
            for actor in self.world.get_actors():
                actor.destroy()


def main():
    ltap_carla_client = LTAPCarlaClient()
    ltap_carla_client.run()


if __name__ == '__main__':
    main()
