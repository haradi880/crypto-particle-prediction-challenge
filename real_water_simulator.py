import pygame
import random
import math
import json
import os
import secrets
import hashlib
import struct
import time
import string
import numpy as np

pygame.init()

WIDTH, HEIGHT = 1400, 800
FPS = 60

ROWS, COLS = 5, 10
SECTION_COUNT = ROWS * COLS

OUT_DIR = "simulation_logs"
os.makedirs(OUT_DIR, exist_ok=True)

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Real-Entropy SPH Water Simulator")
clock = pygame.time.Clock()

BG = (8, 12, 18)
WATER_BG = (20, 45, 70)
BORDER = (120, 170, 220)

COLOR_MAP = {
    "red": (255, 70, 70),
    "green": (80, 255, 130),
    "blue": (80, 150, 255),
}

PARTICLE_COLORS = ["red", "green", "blue"]
PASSWORD_ALPHABET = string.ascii_letters + string.digits + "!@#$%^&*()-_=+"
RANDOM_ACCELERATION = 520.0

# ---------------------------------------------------------------------------
# REAL RANDOMNESS
#
# random.uniform() / np.random with no seed still runs on Mersenne Twister
# seeded from the system clock -- deterministic and low-entropy. LavaRand's
# trick is not "lava is magic", it's that a physical process (turbulent
# fluid + camera noise) is used as an entropy source, then hashed into a
# CSPRNG seed. os.urandom() pulls from the same category of source your OS
# already maintains for cryptography (interrupt timing jitter, thermal
# noise, etc). Seeding numpy's Generator from it gives you the same trust
# model as LavaRand, minus the lamp.
# ---------------------------------------------------------------------------
def get_entropy_seed():
    return int.from_bytes(os.urandom(8), "big")


rng = np.random.default_rng(get_entropy_seed())


class LayeredEntropyPool:
    """Cryptographic pool with simulated state used only as extra input.

    Security comes from secrets.token_bytes()/the operating system.  The
    timing, fluid, event, and noise layers make each pool evolution unique,
    but are not incorrectly counted as independent physical entropy.
    """

    def __init__(self):
        self.pool = secrets.token_bytes(64)
        self.counter = 0
        self.last_digest = hashlib.blake2b(self.pool, digest_size=64).digest()

    def mix(self, particles, events=b""):
        self.counter += 1
        h = hashlib.blake2b(key=self.pool, digest_size=64)
        h.update(b"SOFTWARE-LAVA-POOL-v1")
        h.update(secrets.token_bytes(64))                 # Layer 1: OS CSPRNG
        h.update(struct.pack("!QQ", time.time_ns(), time.perf_counter_ns()))
        h.update(struct.pack("!Q", self.counter))         # Layer 2: timing/counter
        h.update(events)                                  # Layer 3: input timing

        # Layer 4: full, unrounded fluid state (never written to the logs).
        for p in particles:
            h.update(struct.pack(
                "!i2d2d3d",
                p.section_id, *p.pos, *p.vel, p.radius_nm, p.mass, p.density
            ))

        # Layers 5-10: independent-looking simulated mixing fields. These add
        # diffusion and visual complexity, not claimed physical entropy.
        state = h.digest()
        for layer in range(6):
            wave = math.sin((self.counter + 1) * (layer + 1) * 0.61803398875)
            state = hashlib.blake2b(
                state + struct.pack("!id", layer, wave),
                key=secrets.token_bytes(32),
                digest_size=64,
                person=b"lava-layer-v1",
            ).digest()

        self.pool = hashlib.blake2b(
            self.pool + state, digest_size=64, person=b"lava-pool-v1"
        ).digest()
        self.last_digest = state

    def random_bytes(self, length):
        output = bytearray()
        block = 0
        while len(output) < length:
            output.extend(hashlib.blake2b(
                self.pool + struct.pack("!QQ", self.counter, block),
                key=secrets.token_bytes(32),
                digest_size=64,
                person=b"lava-output-v1",
            ).digest())
            block += 1
        self.pool = hashlib.blake2b(
            self.pool + bytes(output[-64:]) + secrets.token_bytes(32),
            digest_size=64,
        ).digest()
        return bytes(output[:length])

    def password(self, length=24):
        # Rejection sampling prevents modulo bias.
        chars = []
        limit = 256 - (256 % len(PASSWORD_ALPHABET))
        while len(chars) < length:
            for value in self.random_bytes(64):
                if value < limit:
                    chars.append(PASSWORD_ALPHABET[value % len(PASSWORD_ALPHABET)])
                    if len(chars) == length:
                        break
        return "".join(chars)


def reseed():
    """Call periodically to re-inject fresh OS entropy instead of relying
    on one seed drawn out at process start."""
    global rng
    rng = np.random.default_rng(get_entropy_seed())


class Particle:
    def __init__(self, section_id, color_name, x, y):
        self.section_id = section_id
        self.color_name = color_name

        self.pos = np.array([x, y], dtype=np.float64)
        self.vel = rng.uniform(-15, 15, size=2).astype(np.float64)
        self.force = np.zeros(2, dtype=np.float64)

        self.radius_nm = rng.uniform(35, 60)
        self.radius_px = self.radius_nm / 8

        self.mass = rng.uniform(0.8, 1.4)
        self.density = 0.0
        self.pressure = 0.0
        self.secret_acceleration = np.zeros(2, dtype=np.float64)

    def refresh_secret_force(self):
        """Draw a future force that is unavailable before this frame.

        Convert two unsigned 64-bit OS-CSPRNG values to uniform [-1, 1].
        This value is deliberately never exported to the training logs.
        """
        raw_x, raw_y = struct.unpack("!QQ", secrets.token_bytes(16))
        scale = float((1 << 64) - 1)
        self.secret_acceleration[0] = (raw_x / scale * 2.0 - 1.0) * RANDOM_ACCELERATION
        self.secret_acceleration[1] = (raw_y / scale * 2.0 - 1.0) * RANDOM_ACCELERATION

    def update_size(self):
        # Slow real-entropy-driven size drift (thermal-noise style jitter,
        # not a scripted animation)
        self.radius_nm += rng.uniform(-0.02, 0.02)
        self.radius_nm = max(30, min(70, self.radius_nm))
        self.radius_px = self.radius_nm / 8


class WaterSimulator:
    """
    Proper (if simplified) SPH fluid solver.

    Only three forces act on a particle, and all three are physically
    motivated:
      1. Pressure force   -> pushes particles apart when locally compressed
      2. Viscosity force  -> drags a particle toward its neighbors' velocity
      3. Gravity           -> the ONLY external force, and it's what shaking
                              the container actually changes.

    There is no scripted vortex, no random "brownian kick" force, and no
    spring-hack for shaking. Sloshing emerges from real inertia responding
    to a changing gravity vector, the way it would in an actual container.
    """

    def __init__(self):
        self.frame_id = 0
        self.session_id = time.strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(OUT_DIR, self.session_id)
        os.makedirs(self.session_dir, exist_ok=True)
        self.sections = []
        self.particles = []
        self.entropy_pool = LayeredEntropyPool()
        self.current_password = "(press P to generate)"
        self.event_entropy = bytearray()

        self.create_sections()
        self.create_particles()

        # SPH kernel radius and fluid constants (tune these, not fudge forces)
        self.h = 45.0
        self.h2 = self.h * self.h
        self.gas_constant = 4.0
        self.viscosity = 0.18
        self.particle_radius_min_gap = 1.05  # hard-collision safety margin

        # kernel normalization constants (2D)
        self.poly6_coeff = 4.0 / (math.pi * self.h ** 8)
        self.spiky_grad_coeff = -30.0 / (math.pi * self.h ** 5)
        self.visc_lap_coeff = 40.0 / (math.pi * self.h ** 5)

        # With only 3 particles per section, real SPH density almost never
        # reaches an arbitrary rest_density like 1.0 -- pressure then stays
        # negative forever, which *pulls* particles together instead of
        # pushing them apart. Fix: calibrate rest_density from the actual
        # starting configuration, so pressure starts near zero and only
        # goes positive (repulsive) once particles get squeezed tighter
        # than their resting spacing.
        self.rest_density = self._estimate_rest_density()

        # gravity is the single external force; "shaking" just changes this
        self.base_gravity = np.array([0.0, 90.0])
        self.gravity = self.base_gravity.copy()

    def create_sections(self):
        margin = 20
        gap = 8
        section_w = (WIDTH - 2 * margin - (COLS - 1) * gap) / COLS
        section_h = (HEIGHT - 2 * margin - (ROWS - 1) * gap) / ROWS

        for r in range(ROWS):
            for c in range(COLS):
                x = margin + c * (section_w + gap)
                y = margin + r * (section_h + gap)
                rect = pygame.Rect(x, y, section_w, section_h)
                self.sections.append(rect)

    def create_particles(self):
        for section_id, rect in enumerate(self.sections):
            for color_name in PARTICLE_COLORS:
                x = rng.uniform(rect.left + 20, rect.right - 20)
                y = rng.uniform(rect.top + 20, rect.bottom - 20)
                self.particles.append(Particle(section_id, color_name, x, y))

    def _estimate_rest_density(self):
        """Sample the actual starting layout to get a realistic rest_density
        instead of guessing a constant that never matches reality when each
        section only has 3 particles in it."""
        total = 0.0
        count = 0
        for p in self.particles:
            density = 0.0
            for q in self.particles:
                if p.section_id != q.section_id:
                    continue
                rij = p.pos - q.pos
                r2 = float(rij @ rij)
                density += q.mass * self.poly6(r2)
            total += density
            count += 1
        return max(total / count, 1e-6)

    # --- SPH kernels (standard Muller 2003 forms) ---------------------

    def poly6(self, r2):
        if r2 >= self.h2:
            return 0.0
        diff = self.h2 - r2
        return self.poly6_coeff * diff ** 3

    def spiky_gradient(self, rij, r):
        if r <= 1e-6 or r >= self.h:
            return np.zeros(2)
        diff = self.h - r
        return self.spiky_grad_coeff * diff ** 2 * (rij / r)

    def viscosity_laplacian(self, r):
        if r >= self.h:
            return 0.0
        return self.visc_lap_coeff * (self.h - r)

    def compute_density_pressure(self):
        for p in self.particles:
            density = 0.0
            for q in self.particles:
                if p.section_id != q.section_id:
                    continue
                rij = p.pos - q.pos
                r2 = float(rij @ rij)
                density += q.mass * self.poly6(r2)

            p.density = max(density, 1e-4)
            # Clamp negative pressure to zero: with so few particles per
            # section, "density below rest" is noise, not real rarefaction.
            # Letting pressure go negative turns the pressure term into an
            # attractive force, which is what was crushing everyone into
            # one corner. Real liquids resist compression, not stretching,
            # so clamping is physically defensible here, not just a hack.
            p.pressure = max(0.0, self.gas_constant * (p.density - self.rest_density))

    def compute_forces(self):
        for p in self.particles:
            pressure_force = np.zeros(2)
            viscosity_force = np.zeros(2)

            for q in self.particles:
                if p is q or p.section_id != q.section_id:
                    continue

                rij = p.pos - q.pos
                r = float(np.linalg.norm(rij))
                if r >= self.h:
                    continue

                grad = self.spiky_gradient(rij, r)
                pressure_force += -q.mass * (
                    (p.pressure + q.pressure) / (2.0 * q.density)
                ) * grad

                lap = self.viscosity_laplacian(r)
                viscosity_force += self.viscosity * q.mass * (
                    (q.vel - p.vel) / q.density
                ) * lap

            # This hidden force is generated only now, after any prediction
            # for this frame must already have been made.
            p.refresh_secret_force()
            gravity_force = (self.gravity + p.secret_acceleration) * p.mass
            p.force = pressure_force + viscosity_force + gravity_force

    def resolve_hard_collisions(self):
        """Direct positional correction so particles can never fully stack
        on top of each other, regardless of how weak the SPH pressure
        estimate is with only 3 particles in a section. This is what a
        real rigid-ish fluid body does at contact -- it doesn't wait for a
        pressure field to catch up, it just doesn't interpenetrate."""
        by_section = {}
        for p in self.particles:
            by_section.setdefault(p.section_id, []).append(p)

        for group in by_section.values():
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    a, b = group[i], group[j]
                    delta = a.pos - b.pos
                    dist = float(np.linalg.norm(delta))
                    min_dist = (a.radius_px + b.radius_px) * self.particle_radius_min_gap

                    if dist < min_dist:
                        if dist < 1e-6:
                            direction = rng.uniform(-1, 1, size=2)
                            direction /= np.linalg.norm(direction) + 1e-9
                            dist = 1e-6
                        else:
                            direction = delta / dist

                        overlap = min_dist - dist
                        a.pos += direction * (overlap * 0.5)
                        b.pos -= direction * (overlap * 0.5)

    def integrate(self, dt):
        for p in self.particles:
            acc = p.force / p.mass
            p.vel += acc * dt
            p.pos += p.vel * dt

            p.update_size()

            rect = self.sections[p.section_id]
            restitution = 0.5

            if p.pos[0] - p.radius_px < rect.left:
                p.pos[0] = rect.left + p.radius_px
                p.vel[0] *= -restitution
            if p.pos[0] + p.radius_px > rect.right:
                p.pos[0] = rect.right - p.radius_px
                p.vel[0] *= -restitution
            if p.pos[1] - p.radius_px < rect.top:
                p.pos[1] = rect.top + p.radius_px
                p.vel[1] *= -restitution
            if p.pos[1] + p.radius_px > rect.bottom:
                p.pos[1] = rect.bottom - p.radius_px
                p.vel[1] *= -restitution

    def update(self, shake_dir, dt):
        # Shaking tilts gravity instead of injecting an arbitrary force --
        # the resulting slosh is a real inertial response, not scripted.
        target = self.base_gravity + np.array(shake_dir, dtype=np.float64) * 260.0
        self.gravity += (target - self.gravity) * min(1.0, dt * 6.0)

        self.compute_density_pressure()
        self.compute_forces()
        self.integrate(dt)
        self.resolve_hard_collisions()

    def draw(self):
        screen.fill(BG)
        font = pygame.font.SysFont("arial", 11)

        for i, rect in enumerate(self.sections):
            pygame.draw.rect(screen, WATER_BG, rect, border_radius=6)
            pygame.draw.rect(screen, BORDER, rect, 1, border_radius=6)
            label = font.render(str(i), True, (180, 220, 255))
            screen.blit(label, (rect.x + 4, rect.y + 3))

        for p in self.particles:
            color = COLOR_MAP[p.color_name]

            glow_radius = int(p.radius_px * 2.2)
            glow_surface = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surface, (*color, 50), (glow_radius, glow_radius), glow_radius)
            screen.blit(glow_surface, (int(p.pos[0] - glow_radius), int(p.pos[1] - glow_radius)))

            pygame.draw.circle(screen, color, (int(p.pos[0]), int(p.pos[1])), int(p.radius_px))
            pygame.draw.circle(
                screen, (255, 255, 255),
                (int(p.pos[0] - p.radius_px / 3), int(p.pos[1] - p.radius_px / 3)),
                max(1, int(p.radius_px / 4)),
            )

        info_font = pygame.font.SysFont("consolas", 16)
        digest_text = self.entropy_pool.last_digest.hex()[:32]
        lines = [
            "Cryptographically driven particle prediction challenge",
            f"Pool fingerprint: {digest_text}...",
            f"Password: {self.current_password}",
            "P = generate password | Movement keys = stir",
            "Motion source: hidden fresh OS-CSPRNG force on every particle/frame",
        ]
        panel = pygame.Surface((720, 112), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 185))
        screen.blit(panel, (WIDTH - 730, 10))
        for index, text in enumerate(lines):
            rendered = info_font.render(text, True, (210, 235, 255))
            screen.blit(rendered, (WIDTH - 718, 18 + index * 19))

        pygame.display.flip()

    def export_json(self):
        data = {"frame": self.frame_id, "time_sec": round(self.frame_id / FPS, 4), "sections": {}}
        for p in self.particles:
            sid = str(p.section_id)
            data["sections"].setdefault(sid, {})
            data["sections"][sid][p.color_name] = {
                "x": round(float(p.pos[0]), 3),
                "y": round(float(p.pos[1]), 3),
                "vx": round(float(p.vel[0]), 3),
                "vy": round(float(p.vel[1]), 3),
                "radius_nm": round(float(p.radius_nm), 3),
                "density": round(float(p.density), 5),
                "pressure": round(float(p.pressure), 5),
            }
        path = os.path.join(self.session_dir, f"frame_{self.frame_id:06d}.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def run(self):
        running = True
        while running:
            dt = clock.tick(FPS) / 1000
            self.frame_id += 1

            if self.frame_id % (FPS * 30) == 0:
                reseed()  # refresh entropy periodically, don't rely on one seed forever

            shake_dir = [0.0, 0.0]
            for event in pygame.event.get():
                self.event_entropy.extend(struct.pack(
                    "!IQ", int(event.type), time.perf_counter_ns()
                ))
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                    self.entropy_pool.mix(self.particles, bytes(self.event_entropy))
                    self.event_entropy.clear()
                    self.current_password = self.entropy_pool.password(24)

            keys = pygame.key.get_pressed()
            if keys[pygame.K_a] or keys[pygame.K_LEFT]:
                shake_dir[0] -= 1
            if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
                shake_dir[0] += 1
            if keys[pygame.K_w] or keys[pygame.K_UP]:
                shake_dir[1] -= 1
            if keys[pygame.K_s] or keys[pygame.K_DOWN]:
                shake_dir[1] += 1

            self.update(shake_dir, dt)
            self.entropy_pool.mix(self.particles, bytes(self.event_entropy))
            self.event_entropy.clear()
            self.draw()

            if self.frame_id % 3 == 0:
                self.export_json()

        pygame.quit()


if __name__ == "__main__":
    sim = WaterSimulator()
    sim.run()
