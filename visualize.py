import matplotlib.pyplot as plt
import matplotlib.patches as patches

from motion.obstacle import ObstacleMap

class Visualizer:
    def __init__(self, omap: ObstacleMap, agents: list):
        # Matplotlib 초기화 (실시간 플로팅 모드)
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        
        # 맵 범위 설정 (PyBullet 카메라 설정과 유사하게)
        self.ax.set_xlim(-100, 300)
        self.ax.set_ylim(-100, 300)
        self.ax.set_aspect('equal') # 1:1 비율 (원이 타원으로 보이지 않게)
        self.ax.grid(True)
        
        self.omap = omap
        self.agents = agents
        
        # 장애물 시각화 
        self._create_obstacles()
        
        # 에이전트 색상
        cmap = plt.colormaps.get_cmap('tab10')
        self.agent_colors = {agent: cmap(i / len(agents)) for i, agent in enumerate(agents)}

        # 에이전트 시각화 객체 (업데이트를 위해 저장)
        self.agent_patches = {}     # 에이전트 동체 (plt.Circle)
        self.target_markers = {}    # 에이전트 목표 (plt.plot)
        self.gbp_traj_lines = {}    # GBP 궤적 (plt.plot)
        self.mppi_traj_lines = {}   # MPPI 궤적 (plt.plot)
        
        self._create_agents()
        
        self.ax.legend(loc='upper right')

    def _create_obstacles(self):
        """장애물을 Matplotlib Patch로 생성"""
        for name, obj in self.omap.objects.items():
            if obj['type'] == 'circle':
                circle = patches.Circle(
                    (obj['centerx'], obj['centery']),
                    obj['radius'],
                    color='black',
                    alpha=0.8,
                    label='Obstacle'
                )
                self.ax.add_patch(circle)
    
    def _create_agents(self):
        """에이전트 아티스트(Patch, Line)들을 생성"""
        for agent in self.agents:
            color = self.agent_colors[agent]
            
            # 1. 에이전트 동체 (Circle Patch)
            agent_patch = patches.Circle(
                (agent.x, agent.y),
                agent.r,
                color=color,
                alpha=0.9,
                label=f'Agent {agent.name}'
            )
            self.ax.add_patch(agent_patch)
            self.agent_patches[agent] = agent_patch
            
            # 2. 목표 지점 (X 마커)
            target = agent.get_target()
            target_marker, = self.ax.plot(
                target[0, 0], 
                target[1, 0], 
                marker='x', 
                markersize=10, 
                color=color,
                mew=2 # 마커 선 굵기
            )
            self.target_markers[agent] = target_marker
            
            # 3. GBP 궤적 (초기에는 빈 데이터로 생성)
            gbp_line, = self.ax.plot([], [], color=color, linewidth=2)
            self.gbp_traj_lines[agent] = gbp_line
            
            # 4. MPPI 궤적 (점선, 더 밝은 색)
            color_mppi = [min(1.0, c + 0.3) for c in color[:3]] # 밝은 색
            mppi_line, = self.ax.plot(
                [], [], 
                color=color_mppi, 
                linewidth=1.5, 
                linestyle='--'
            )
            self.mppi_traj_lines[agent] = mppi_line

    def update_visualization(self):
        """에이전트 위치 및 궤적 데이터 업데이트"""
        for agent in self.agents:
            patch = self.agent_patches[agent]
            gbp_line = self.gbp_traj_lines[agent]
            mppi_line = self.mppi_traj_lines[agent]
            
            state = agent.get_state()
            
            # 1. 에이전트 위치 업데이트
            if state[0] is not None:
                x, y, _, _ = state[0]
                patch.center = (x, y)
            
            # 2. GBP 궤적 업데이트
            #   state 리스트에서 x, y 좌표만 추출
            gbp_x = [s[0] for s in state if s is not None]
            gbp_y = [s[1] for s in state if s is not None]
            gbp_line.set_data(gbp_x, gbp_y)
            
            # 3. MPPI 궤적 업데이트
            mppi_traj = agent.get_trajectory()
            if mppi_traj is not None:
                mppi_x = [s[0, 0] for s in mppi_traj]
                mppi_y = [s[1, 0] for s in mppi_traj]
                mppi_line.set_data(mppi_x, mppi_y)
            else:
                mppi_line.set_data([], []) # 궤적이 없으면 비움
        
        # 캔버스 다시 그리기
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001) # GUI가 업데이트될 시간을 줌
    
    def close(self):
        plt.ioff() # 실시간 모드 끄기
        plt.close(self.fig)
        print("Matplotlib visualizer closed.")