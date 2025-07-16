from shapely.geometry import Point, Polygon
import time, math, cv2, os, time, threading

class Tracker:
    def __init__(self, _main, _worker, tracker_dimensions, buffer_dimensions, log, distorted_factor, solver_data, vehicle_direction=1):
        self.tracker_dimensions = tracker_dimensions
        self.buffer_dimensions = buffer_dimensions
        self.distorted_factor = distorted_factor
        self.solver_data = solver_data
        self.log = log

        self.vehicle_direction = vehicle_direction
        self.tracker_start_limit = []
        self.tracker_stop_limit = []

        if self.vehicle_direction == 0:
            self.tracker_start_limit.append(buffer_dimensions[1])
            self.tracker_start_limit.append(0)
            self.tracker_stop_limit.append(self.tracker_dimensions[1])
            self.tracker_stop_limit.append(self.tracker_dimensions[1] - (self.buffer_dimensions[1]+20))
        elif self.vehicle_direction == 1:
            self.tracker_start_limit.append(self.tracker_dimensions[1])
            self.tracker_start_limit.append(self.tracker_dimensions[1] - (self.buffer_dimensions[1]))
            self.tracker_stop_limit.append(buffer_dimensions[1]+20)
            self.tracker_stop_limit.append(0)


        self._main = _main
        self._worker =_worker

        self.items = []
        self.current_points = []
        self.prev_points = []
        self.unique_id_counter = 0

        self.filedir = os.path.join("records", str(self._main.startTime).replace(" ", "_").replace(":", "-").split(".")[0])
        self.imgdir = os.path.join(self.filedir, f"images-lane-{str(self._worker)}")
        self.logdir = os.path.join(self.filedir, f"lane-{str(self._worker)}_log.csv")

        if not os.path.exists(self.filedir): os.mkdir(self.filedir)
        if not os.path.exists(self.imgdir): os.mkdir(self.imgdir)
        if not os.path.exists(self.logdir): 
            with open(self.logdir, "w") as log:
                log.write("id,vehicle_type,state,abs_vehicle_speed,imgname,start_pos,end_pos,start_time,end_time,\n")
        

    def in_region(self, polygon, point=()):
        check_point = Point(point[0], point[1])
        return polygon.contains(check_point)

    def track(self, vehicles, speed_limits):
        self.current_vehicles = vehicles[:]
        new_items = []
        lost_ids = [item[0] for item in self.prev_points]

        for vehicles in self.current_vehicles:
            new_point = Point(vehicles[1][0], vehicles[1][1])
            found_match = False

            for old_point in self.prev_points:
                old_id, old_coords = old_point
                old_point = Point(old_coords[0], old_coords[1])

                # Create a polygon (around the old point)
                width = self.buffer_dimensions[0]
                height = self.buffer_dimensions[1]

                polygon = Polygon([ (old_coords[0] - width/2, old_coords[1] - height/2),
                                    (old_coords[0] + width/2, old_coords[1] - height/2),
                                    (old_coords[0] + width/2, old_coords[1] + height/2),
                                    (old_coords[0] - width/2, old_coords[1] + height/2),])

                if self.in_region(polygon, new_point.coords[0]):
                    try:    
                        self.log[str(old_id)][1] = new_point.coords[0]
                        self.log[str(old_id)][3] = time.time()
                    except:pass
                    try:
                        if (self.tracker_stop_limit[1] < new_point.coords[0][1] < self.tracker_stop_limit[0]):
                            thread_solver = threading.Thread(target=self.solver, args=[old_id, self.current_vehicles, self.log[str(old_id)], self.solver_data, speed_limits])
                            thread_solver.start()
                            del(self.log[str(old_id)])
                        else:
                            new_items.append((old_id, new_point.coords[0]))
                    except:pass

                    found_match = True
                    while old_id in lost_ids:
                        lost_ids.remove(old_id)      
                    break

            if not found_match and (self.tracker_start_limit[1] < new_point.coords[0][1] < self.tracker_start_limit[0]):
                new_items.append((self.unique_id_counter, new_point.coords[0]))
                # log - (starting point, current point, starting time, current time)
                self.log[str(self.unique_id_counter)] = [new_point.coords[0], new_point.coords[0], time.time(), time.time()]

                _, vehicle_img = self.find_vehicle(self.log[str(self.unique_id_counter)], self.current_vehicles)
                imgname = time.strftime("%Y%m%d-%H%M%S")
                cv2.imwrite(filename=os.path.join(self.imgdir, f"{self.unique_id_counter}_{imgname}.jpg"), img=vehicle_img)

                self.unique_id_counter += 1

        if lost_ids:
            for id in lost_ids:
                try:
                    print(f"Tracker {self._worker} : ID lost: {id}")
                    state = "lost"
                    thread_solver = threading.Thread(target=self.solver, args=[id, self.prev_vehicles, self.log[str(id)], self.solver_data, speed_limits, state])
                    thread_solver.start()
                
                    while str(id) in self.log:
                        del(self.log[str(id)])
                except:pass

        self.items = new_items
        self.prev_points = self.items
        self.prev_vehicles = self.current_vehicles
        return self.items


    def solver(self, id, vehicles, record, solver_data, speed_limits, state="success"):
        try:
            pos_distance = math.dist(record[0], record[1])
            time_diff = record[3]-record[2]

            data_pos_distance = math.dist(solver_data[0][0], solver_data[0][1])
            data_abs_distance = solver_data[1]
            distance_ratio = data_abs_distance/data_pos_distance

            abs_vehicle_displacement = distance_ratio*pos_distance
            abs_vehicle_speed = abs_vehicle_displacement/time_diff
        except:
            abs_vehicle_speed = "error"

        vehicle_type, vehicle_img = self.find_vehicle(record, vehicles)

        imgname = time.strftime("%Y%m%d-%H%M%S")
        cv2.imwrite(filename=os.path.join(self.imgdir, f"{id}_{imgname}.jpg"), img=vehicle_img)

        with open(self.logdir, "+a") as log:
            record[0] = str(record[0]).replace(","," ")
            record[1] = str(record[1]).replace(","," ")
            log.write(f"{id},{vehicle_type},{state},{abs_vehicle_speed},{id}_{imgname}.jpg,{record[0]},{record[1]},{record[2]},{record[3]},\n")


    def find_vehicle(self, record, vehicles):
        vehicle_type = None
        for item in vehicles:
            if record[1] == item[1]:
                vehicle_type = item[0]
                vehicle_img = item[3]
        return vehicle_type, vehicle_img