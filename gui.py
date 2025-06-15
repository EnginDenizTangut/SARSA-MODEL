import sys
import numpy as np
import random
from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout, QLabel, QPushButton
from PyQt5.QtGui import QColor, QPalette
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject

# --- Ortam ve Robot Ayarları ---
GRID_SIZE = 10  # Izgara boyutu (örn. 10x10)
START_STATE = (0, 0)  # Başlangıç noktası
TARGET_STATE = (GRID_SIZE - 1, GRID_SIZE - 1)  # Hedef noktası (sağ alt köşe)
NUM_OBSTACLES = 20  # Rastgele engel sayısı

# Eylemler: Yukarı, Aşağı, Sol, Sağ
ACTIONS = {
    0: "UP",
    1: "DOWN",
    2: "LEFT",
    3: "RIGHT"
}
NUM_ACTIONS = len(ACTIONS)

# Ödüller
REWARD_GOAL = 100
REWARD_OBSTACLE = -50
REWARD_MOVE = -1

# --- SARSA Parametreleri ---
ALPHA = 0.1       # Öğrenme oranı
GAMMA = 0.9       # İskonto faktörü
EPSILON_START = 1.0  # Keşif oranı başlangıç değeri
EPSILON_END = 0.01  # Keşif oranı bitiş değeri
EPSILON_DECAY = 0.999  # Keşif oranı azalma faktörü

# --- SARSA Agent Sınıfı ---
class SARSAAgent(QObject):
    # GUI'ye durum ve ödül bilgisini göndermek için sinyal
    update_gui_signal = pyqtSignal(tuple, int, list, str)

    def __init__(self, grid_size, start_state, target_state, num_obstacles):
        super().__init__()
        self.grid_size = grid_size
        self.start_state = start_state
        self.target_state = target_state
        self.num_obstacles = num_obstacles

        self.q_table = np.zeros((self.grid_size * self.grid_size, NUM_ACTIONS))
        self.epsilon = EPSILON_START
        self.episode_count = 0
        self.steps_in_episode = 0

        self.current_grid, self.obstacles = self._create_grid_and_obstacles()
        self.current_state = self.start_state
        self.current_action = self._choose_action(self.current_state, self.epsilon)
        self.path_taken = [self.current_state]

    def _create_grid_and_obstacles(self):
        """
        Izgara oluşturur ve rastgele engeller yerleştirir.
        Engel, başlangıç ve hedef çakışmalarını önler.
        """
        grid = np.zeros((self.grid_size, self.grid_size))
        obstacles = set()
        while len(obstacles) < self.num_obstacles:
            r, c = random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)
            if (r, c) != self.start_state and (r, c) != self.target_state:
                obstacles.add((r, c))
                grid[r, c] = -1  # Engelleri -1 ile işaretle
        return grid, obstacles

    def _get_state_index(self, state):
        """Durum koordinatlarını tekil bir indekse dönüştürür."""
        return state[0] * self.grid_size + state[1]

    def _get_next_state_reward_done(self, current_state, action):
        """
        Verilen eyleme göre bir sonraki durumu, ödülü ve oyunun bitip bitmediğini döndürür.
        """
        r, c = current_state

        if action == 0:  # UP
            r_new, c_new = r - 1, c
        elif action == 1:  # DOWN
            r_new, c_new = r + 1, c
        elif action == 2:  # LEFT
            r_new, c_new = r, c - 1
        elif action == 3:  # RIGHT
            r_new, c_new = r, c + 1

        # Sınır kontrolü
        if not (0 <= r_new < self.grid_size and 0 <= c_new < self.grid_size):
            return current_state, REWARD_OBSTACLE, True  # Duvara çarpma ceza ve bölüm biter

        next_state = (r_new, c_new)

        if next_state == self.target_state:
            return next_state, REWARD_GOAL, True  # Hedefe ulaşıldı
        elif next_state in self.obstacles:
            return next_state, REWARD_OBSTACLE, True  # Engele çarpma
        else:
            return next_state, REWARD_MOVE, False  # Normal hareket

    def _choose_action(self, state, epsilon):
        """
        Epsilon-greedy stratejisi ile bir eylem seçer.
        """
        if random.uniform(0, 1) < epsilon:
            return random.randint(0, NUM_ACTIONS - 1)  # Keşif: Rastgele eylem seç
        else:
            # Sömürü: Q-tablosundaki en iyi eylemi seç
            state_idx = self._get_state_index(state)
            return np.argmax(self.q_table[state_idx, :])

    def reset_episode(self):
        """Yeni bir bölüm başlatır."""
        self.current_state = self.start_state
        self.current_action = self._choose_action(self.current_state, self.epsilon)
        self.steps_in_episode = 0
        self.path_taken = [self.current_state]
        self.episode_count += 1
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)  # Epsilon'ı azalt
        self.update_gui_signal.emit(self.current_state, REWARD_MOVE, self.path_taken, f"Bölüm {self.episode_count} başladı.")

    def step(self):
        """SARSA algoritmasında bir adım ilerler."""
        if self.steps_in_episode > self.grid_size * self.grid_size * 2:  # Sonsuz döngü önleme
            self.reset_episode()
            return

        state_idx = self._get_state_index(self.current_state)

        # Eylemi gerçekleştir ve gözlemleri al
        next_state, reward, done = self._get_next_state_reward_done(self.current_state, self.current_action)

        # Bir sonraki durumu indeksine dönüştür
        next_state_idx = self._get_state_index(next_state)

        # Bir sonraki eylemi seç (SARSA'nın ana noktası)
        if not done:
            next_action = self._choose_action(next_state, self.epsilon)
            td_target = reward + GAMMA * self.q_table[next_state_idx, next_action]
        else:
            td_target = reward  # Terminal durumunda gelecekte ödül yok
            next_action = None  # Terminal durumdan sonra eylem yok

        # Q-değerini güncelle
        self.q_table[state_idx, self.current_action] = self.q_table[state_idx, self.current_action] + ALPHA * (td_target - self.q_table[state_idx, self.current_action])

        # Durumu ve eylemi güncelle
        self.current_state = next_state
        if not done:
            self.current_action = next_action  # Bir sonraki adımda yapılacak eylem

        self.path_taken.append(self.current_state)
        self.steps_in_episode += 1

        status_message = ""
        if done:
            if self.current_state == self.target_state:
                status_message = f"Hedefe ulaşıldı! Bölüm {self.episode_count} bitti."
            elif self.current_state in self.obstacles:
                status_message = f"Engele çarpıldı! Bölüm {self.episode_count} bitti."
            else:  # Duvara çarpma
                status_message = f"Duvara çarpıldı! Bölüm {self.episode_count} bitti."

            self.update_gui_signal.emit(self.current_state, reward, self.path_taken, status_message)
            self.reset_episode()  # Yeni bir bölüm başlat
        else:
            self.update_gui_signal.emit(self.current_state, reward, self.path_taken, f"Adım: {self.steps_in_episode}, Ödül: {reward}")

# --- Ana GUI Penceresi Sınıfı ---
class RobotSARSAApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SARSA Robot Simülasyonu")
        self.setGeometry(100, 100, 800, 800)  # Pencere boyutu

        self.grid_labels = {}

        # SARSAAgent nesnesini init_ui() çağrısından önce oluştur.
        self.agent = SARSAAgent(GRID_SIZE, START_STATE, TARGET_STATE, NUM_OBSTACLES)

        self.init_ui()

        # SARSAAgent'tan gelen sinyalleri GUI güncelleme metoduna bağla
        self.agent.update_gui_signal.connect(self.update_grid_ui)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.agent.step)  # Her timer tetiklendiğinde agent'ın adım atmasını sağlar
        self.timer_interval = 100  # ms, animasyon hızı

        self.start_simulation()

    def init_ui(self):
        self.main_layout = QGridLayout()
        self.setLayout(self.main_layout)

        # Izgara oluşturma
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                label = QLabel()
                label.setFixedSize(50, 50)  # Hücre boyutu
                label.setAlignment(Qt.AlignCenter)
                label.setStyleSheet("border: 1px solid black;")
                self.main_layout.addWidget(label, r, c)
                self.grid_labels[(r, c)] = label

        # Kontrol butonları ve durum etiketi
        self.start_button = QPushButton("Başlat")
        self.start_button.clicked.connect(self.start_simulation)
        self.main_layout.addWidget(self.start_button, GRID_SIZE, 0)

        self.stop_button = QPushButton("Durdur")
        self.stop_button.clicked.connect(self.stop_simulation)
        self.main_layout.addWidget(self.stop_button, GRID_SIZE, 1)

        self.speed_up_button = QPushButton("Hızlandır")
        self.speed_up_button.clicked.connect(self.speed_up_simulation)
        self.main_layout.addWidget(self.speed_up_button, GRID_SIZE, 2)

        self.slow_down_button = QPushButton("Yavaşlat")
        self.slow_down_button.clicked.connect(self.slow_down_simulation)
        self.main_layout.addWidget(self.slow_down_button, GRID_SIZE, 3)

        self.status_label = QLabel("Simülasyon Hazır.")
        self.main_layout.addWidget(self.status_label, GRID_SIZE + 1, 0, 1, GRID_SIZE)  # Tüm genişliği kapla

        self.draw_initial_grid()

    def draw_initial_grid(self):
        # Tüm hücreleri varsayılan renge sıfırla
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                label = self.grid_labels[(r, c)]
                label.setText("")
                label.setStyleSheet("background-color: lightgray; border: 1px solid black;")

        # Engelleri çiz
        for r, c in self.agent.obstacles:
            label = self.grid_labels[(r, c)]
            label.setStyleSheet("background-color: darkred; border: 1px solid black;")  # Engel rengi
            label.setText("X")
            label.setFont(label.font())  # Fontu tekrar ayarla

        # Hedefi çiz
        label_target = self.grid_labels[TARGET_STATE]
        label_target.setStyleSheet("background-color: lightgreen; border: 1px solid black;")  # Hedef rengi
        label_target.setText("T")
        label_target.setFont(label_target.font())

        # Başlangıç noktasını çiz
        label_start = self.grid_labels[START_STATE]
        label_start.setStyleSheet("background-color: lightblue; border: 1px solid black;")  # Başlangıç rengi
        label_start.setText("S")
        label_start.setFont(label_start.font())

    def update_grid_ui(self, current_pos, reward, path_taken, status_message):
        self.draw_initial_grid()  # Önce tüm hücreleri sıfırla ve sabitleri çiz

        # Robotun gittiği yolu işaretle
        for r, c in path_taken:
            if (r, c) != START_STATE and (r, c) != TARGET_STATE and (r, c) not in self.agent.obstacles:
                label = self.grid_labels[(r, c)]
                label.setStyleSheet("background-color: lightyellow; border: 1px solid black;")  # Yol rengi
                label.setText("*")

        # Robotun mevcut konumunu çiz
        label_robot = self.grid_labels[current_pos]
        label_robot.setStyleSheet("background-color: orange; border: 1px solid black;")  # Robot rengi
        label_robot.setText("R")
        label_robot.setFont(label_robot.font())

        self.status_label.setText(f"{status_message} | Bölüm: {self.agent.episode_count} | Epsilon: {self.agent.epsilon:.3f}")

    def start_simulation(self):
        self.timer.start(self.timer_interval)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status_label.setText("Simülasyon Başladı...")

    def stop_simulation(self):
        self.timer.stop()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_label.setText("Simülasyon Durduruldu.")

    def speed_up_simulation(self):
        if self.timer_interval > 10:
            self.timer_interval -= 10
            self.timer.setInterval(self.timer_interval)

    def slow_down_simulation(self):
        self.timer_interval += 10
        self.timer.setInterval(self.timer_interval)

# --- Uygulamayı Çalıştır ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RobotSARSAApp()
    window.show()
    sys.exit(app.exec_())