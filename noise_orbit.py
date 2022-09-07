import enum
import dataclasses
from typing import Callable, Optional, List, Tuple

import matplotlib.cm as cm

import cv2
import numpy as np
from tqdm.auto import tqdm

from perlin import PerlinNoiseFactory


def to_bgr(mpl_color: np.ndarray) -> np.ndarray:
    """
    Convert matplotlib color code to BGR for OpenCV
    """
    return tuple((255 * np.array(mpl_color[:3][::-1])).astype(np.int32).tolist())


class Action(enum.Enum):
    ACC = 0
    TURN_RIGHT = 1
    TURN_LEFT = 2


@dataclasses.dataclass
class CarMetrics:
    timestamp: float = 0.  # seconds
    rpm: float = 0.  # 1
    drpm: float = 0.  # 1/s
    g_forw: float = 0.  # m/s^2
    g_rad: float = 0.  # m/s^2
    pos_x: float = 0.
    pos_y: float = 0.
    pos_z: float = 0.
    pos_x2: float = 0.
    pos_y2: float = 0.
    pos_z2: float = 0.
    speed: float = 0.  # km/h


@dataclasses.dataclass
class NoiseOrbit:
    width: float = 400.
    height: float = 400.
    x0: float = .5
    y0: float = .5
    amp: float = 0.09
    background: Optional[np.ndarray] = None
    noise_p: Callable = PerlinNoiseFactory(dimension=3)

    noise_period: float = 180.
    rpm_factor: float = 0.
    last_updated_frame: int = 0
    bg_old: Optional[np.ndarray] = None
    bg_new: Optional[np.ndarray] = None
    trajectory: List[Tuple[float, float, float]] = dataclasses.field(default_factory=lambda: [])
    speed: List[float] = dataclasses.field(default_factory=lambda: [])
    trajectory2: List[Tuple[float, float, float]] = dataclasses.field(default_factory=lambda: [])
    speed2: List[float] = dataclasses.field(default_factory=lambda: [])
    backgrounds: List = dataclasses.field(default_factory=lambda: [
        'mono/pics/city2.jpg',
    ])

    @staticmethod
    def scale(val: float, k: Optional[float]) -> float:
        if k is None:
            return val
        return k * val
    
    def init(self):
        self.x_traj_min = min([p[0] for p in self.trajectory])
        self.x_traj_max = max([p[0] for p in self.trajectory])
        self.y_traj_min = min([p[1] for p in self.trajectory])
        self.y_traj_max = max([p[1] for p in self.trajectory])

        self.backgrounds = [
            cv2.imread('mono/pics/city2.jpg'),
            cv2.imread('mono/pics/city2.jpg'),
            cv2.imread('mono/pics/forest.jpg'),
            cv2.imread('mono/pics/forest.jpg'),
            cv2.imread('mono/pics/sea.jpg'),
        ]
        for ch in range(3):
            self.backgrounds[1][:, :, ch] = cv2.cvtColor(self.backgrounds[0], cv2.COLOR_BGR2GRAY)
            self.backgrounds[3][:, :, ch] = cv2.cvtColor(self.backgrounds[2], cv2.COLOR_BGR2GRAY)
        
        for b_id in range(2, len(self.backgrounds)):
            self.backgrounds[b_id] = cv2.resize(
                self.backgrounds[b_id], self.backgrounds[0].shape[:2][::-1]
            )

    def w(self, k: Optional[float]) -> float:
        return NoiseOrbit.scale(self.width, k)
    
    def h(self, k: Optional[float]) -> float:
        return NoiseOrbit.scale(self.height, k)

    def make_circle(self, num_sides, radius):
        return np.array([
            (radius * np.cos(theta), radius * np.sin(theta))
            for theta in np.arange(0, 2*np.pi, 2*np.pi / num_sides)
        ])
    
    def distort_polygon(self, points: np.ndarray, frame_count) -> np.ndarray:
        points_distorted = np.zeros_like(points)
        for i, (x, y) in enumerate(points):
            new_x, new_y = self.noise(x, y, frame_count)
            points_distorted[i, 0] = new_x
            points_distorted[i, 1] = new_y

        return points_distorted
    
    def noise(self, x, y, frame_count):
        z = frame_count / self.noise_period
        z2 = 0
        distance = np.sqrt((x - self.x0) ** 2 + (y - self.y0) ** 2)

        noise_x = x * distance * 2 + z2
        noise_y = y * distance * 2 + z2

        theta = self.noise_p(noise_x, noise_y, z) * np.pi * 3
        amount_to_nudge = (self.amp + self.rpm_factor) * (1 - np.cos(z))
        new_x = x + amount_to_nudge * np.cos(theta)
        new_y = y + amount_to_nudge * np.sin(theta)

        return new_x, new_y

    def compose_frame(self, frame_count, metrics: CarMetrics):
        self.update_params(metrics)
        frame = self.get_background(frame_count).copy()
        self.width, self.height = frame.shape[:2]

        # add trajectory
        if frame_count % 250 < 300:
            frame = self.draw_trajectory(metrics, frame, frame_count)

        # add circles
        bg = np.zeros_like(frame)
        circles = []
        for radius in np.arange(0.01, 1.5, 0.02):
            points = self.make_circle(80, radius)
            points = self.distort_polygon(points, frame_count)
            circles.append(points)

        for points in circles:
            rad = min(self.height, self.width)
            points_px = np.array([
                (round(self.height * self.y0 + rad * p[0]), round(self.width * self.x0 + rad * p[1]))
                for p in points
            ], np.int32).reshape((-1, 1, 2))
            cv2.polylines(bg, [points_px], True, (255, 255, 255), 1)

        frame[bg > 250] = frame[bg > 250] * 0.7

        return frame
    
    def get_background(self, frame_count):
        return self.backgrounds[0]


    def draw_trajectory(self, metrics, frame, frame_count, duration_frames=90):
        x = np.array([p[0] for p in self.trajectory[:frame_count]])
        y = np.array([p[1] for p in self.trajectory[:frame_count]])
        if len(x) < 5:  # prevent drawing too few points
            return frame

        x = x - self.x_traj_min
        x = self.width * (0.1 + x / self.x_traj_max * 0.8)
        y = y - self.y_traj_min
        y = self.height * (0.1 + (1 - y / self.y_traj_max) * 0.8)
        speed = np.array(self.speed[:frame_count])
        c = [cm.jet(sp / 300) for sp in speed]
        s = 1 + np.clip(speed, 1, 200) / 200 * 4

        traj_frame = np.zeros_like(frame)
        for color, size, xp, yp in zip(c, s, x, y):
            cv2.circle(traj_frame, (int(yp), int(xp)), int(size), to_bgr(color), int(size))
        

        traj_mask = traj_frame.sum(axis=2) > 10
        frame[traj_mask] = 0.3 * frame[traj_mask] + 0.7 * traj_frame[traj_mask]

        ###########################
        # 2nd trajectory
        ###########################
        x = np.array([p[0] for p in self.trajectory[:frame_count]])
        y = np.array([p[1] for p in self.trajectory[:frame_count]])
        if len(x) < 5:  # prevent drawing too few points
            return frame

        x = x - self.x_traj_min
        x = self.width * (0.1 + x / self.x_traj_max * 0.8) - 10
        y = y - self.y_traj_min
        y = self.height * (0.1 + (1 - y / self.y_traj_max) * 0.8) + 10
        speed = np.array(self.speed[:frame_count])
        c = [cm.summer(sp / 300) for sp in speed]
        s = 1 + np.clip(speed, 1, 200) / 200 * 2

        traj_frame = np.zeros_like(frame)
        for color, size, xp, yp in zip(c, s, x, y):
            cv2.circle(traj_frame, (int(yp), int(xp)), int(size), to_bgr(color), int(size))
        

        traj_mask = traj_frame.sum(axis=2) > 10
        frame[traj_mask] = 0.3 * frame[traj_mask] + 0.7 * traj_frame[traj_mask]

        return frame

    def get_porsche_moment(self, metrics) -> Optional[Action]:
        if metrics.g_forw > 13.:  # m/s^2
            return Action.ACC

        return None

    def update_params(self, metrics: CarMetrics):
        drpm_max = 25000  # 1/s
        self.rpm_factor = 0.03 * max(metrics.drpm, 0) / drpm_max

        if abs(metrics.g_forw) < 5:
            self.x0 += (0.5 - self.x0) * 0.2
        else:
            self.x0 += metrics.g_forw * 0.0003
            self.x0 = np.clip(self.x0, 0.1, 0.9)

        if abs(metrics.g_rad) < 5:
            self.y0 += (0.5 - self.y0) * 0.5
        else:
            self.y0 += metrics.g_rad * 0.5
            self.y0 = np.clip(self.y0, 0.1, 0.9)


if __name__ == '__main__':
    engine = NoiseOrbit(width=1000, height=1000)
    metrics = CarMetrics()
    n_frames = 1000
    for frame_count in tqdm(range(n_frames)):
        frame = engine.compose_frame(frame_count, metrics)

        cv2.imshow('Test', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    print('Ok')
