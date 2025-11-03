
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import time

from motion.obstacle import ObstacleMap
from motion.agent import Agent, Env


class MatplotlibVisualizer:
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
                    color='gray',
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


if __name__ == '__main__':
    omap = ObstacleMap()
    omap.set_circle('obstacle1', 100, 100, 20)
    
    agent0 = Agent('a0', [0, 0, 0, 0], [200, 200, 0, 0], steps=8, radius=5, omap=omap)
    agent1 = Agent('a1', [0, 200, 0, 0], [200, 0, 0, 0], steps=8, radius=5, omap=omap)
    agent2 = Agent('a2', [200, 200, 0, 0], [0, 0, 0, 0], steps=8, radius=5, omap=omap)
    agent3 = Agent('a3', [200, 0, 0, 0], [0, 200, 0, 0], steps=8, radius=5, omap=omap)
    agent4 = Agent('a4', [100, 0, 0, 0], [100, 200, 0, 0], steps=8, radius=5, omap=omap)
    agent5 = Agent('a5', [0, 100, 0, 0], [200, 100, 0, 0], steps=8, radius=5, omap=omap)
    agent6 = Agent('a6', [200, 100, 0, 0], [0, 100, 0, 0], steps=8, radius=5, omap=omap)
    agent7 = Agent('a7', [100, 200, 0, 0], [100, 0, 0, 0], steps=8, radius=5, omap=omap)
    
    # 환경 설정
    env = Env()
    env.add_agent(agent0)
    env.add_agent(agent1)
    env.add_agent(agent2)
    env.add_agent(agent3)
    env.add_agent(agent4)
    env.add_agent(agent5)
    env.add_agent(agent6)
    env.add_agent(agent7)
    
    agents = [agent0, agent1, agent2, agent3, agent4, agent5, agent6, agent7]
    
    visualizer = MatplotlibVisualizer(omap, agents)
    
    print("Matplotlib 시뮬레이션 시작...")
    print("종료: Ctrl+C 또는 창 닫기")
    
    step_count = 0
    max_steps = 200
    
    try:
        while step_count < max_steps:
            # GBP 기반 경로 계획
            env.step_plan(iters=5)
            
            # 시각화 업데이트
            visualizer.update_visualization()
            
            # 에이전트 이동
            env.step_move()
            
            # 진행 상황 출력
            if step_count % 10 == 0:
                print(f"Step {step_count}/{max_steps}")
                all_reached_in_step = True
                for agent in agents:
                    state = agent.get_state()
                    if state[0] is not None:
                        x, y, vx, vy = state[0]
                        target = agent.get_target()
                        dist_to_goal = np.linalg.norm([x - target[0, 0], y - target[1, 0]])
                        print(f"  {agent.name}: pos=({x:.1f}, {y:.1f}), dist_to_goal={dist_to_goal:.1f}")
                        if dist_to_goal > 10:
                            all_reached_in_step = False
                    else:
                        all_reached_in_step = False # None이면 아직 도달 못한 것
                
                if all_reached_in_step:
                    print("\n(출력) 모든 에이전트가 목표 근처 도달 시작")
    
            time.sleep(0.001) 
            
            step_count += 1
            
            # 모든 에이전트가 목표에 도달했는지 확인
            all_reached = True
            for agent in agents:
                state = agent.get_state()
                if state[0] is not None: # 아직 움직일 상태가 남아있다면
                    x, y, _, _ = state[0]
                    target = agent.get_target()
                    dist = np.linalg.norm([x - target[0, 0], y - target[1, 0]])
                    if dist > 10:  
                        all_reached = False
                        break
                else: 
                    pass 
            
            if all_reached:
                print("\n모든 에이전트가 목표에 도달했습니다!")
                print("\n10초후에 시뮬레이션을 종료합니다.")
                plt.pause(10) # 10초간 마지막 모습 보여주기
                break
                
    except KeyboardInterrupt:
        print("\n시뮬레이션 중단됨")
    
    finally:
        print("시뮬레이션 종료")
        visualizer.close()