import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import time
from matplotlib.animation import PillowWriter

from motion.obstacle import ObstacleMap
from motion.agent import Agent, Env
from visualize import Visualizer

if __name__ == '__main__':
    # 장애물 맵 생성
    omap = ObstacleMap(safe_margin=15)
    
    # 가운데 세로 벽 (위쪽 부분) - 좁은 통로를 위해 두 개의 직사각형으로 분리
    # 전체 높이 200, 통로 위치 y=100, 통로 폭 30
    wall_width = 100
    passage_width = 50
    passage_center = 100
    
    # 위쪽 벽 (y = 100+15 ~ 200)
    upper_wall_height = 400 - (passage_center + passage_width/2)
    upper_wall_y = passage_center + passage_width/2 + upper_wall_height/2
    omap.set_rectangle('wall_upper', 100, upper_wall_y, wall_width, upper_wall_height, theta=0.0)
    
    # 아래쪽 벽 (y = 0 ~ 100-15)
    lower_wall_height = passage_center - passage_width/2 - 400
    lower_wall_y = lower_wall_height/2 + (passage_center - passage_width/2)
    omap.set_rectangle('wall_lower', 100, lower_wall_y, wall_width, lower_wall_height, theta=0.0)
    
    # 왼쪽 에이전트들 (x=20, y 분포)
    agent0 = Agent('L0', [-50, 70, 0, 0], [250, 70, 0, 0], steps=12, radius=5, omap=omap)
    agent1 = Agent('L1', [-50, 100, 0, 0], [250, 100, 0, 0], steps=12, radius=5, omap=omap)
    agent2 = Agent('L2', [-50, 130, 0, 0], [250, 130, 0, 0], steps=12, radius=5, omap=omap)

    
    # 오른쪽 에이전트들 (x=180, y 분포)
    agent3 = Agent('R0', [250, 70, 0, 0], [-50, 70, 0, 0], steps=12, radius=5, omap=omap)
    agent4 = Agent('R1', [250, 100, 0, 0], [-50, 100, 0, 0], steps=12, radius=5, omap=omap)
    agent5 = Agent('R2', [250, 130, 0, 0], [-50, 130, 0, 0], steps=12, radius=5, omap=omap)

    
    # 환경 설정
    env = Env()
    agents = [agent0, agent1, agent2, agent3, agent4, agent5]
    for a in agents:
        env.add_agent(a)
    
    visualizer = Visualizer(omap, agents, xlim=(-100, 300), ylim=(-100, 300))


    print("종료: Ctrl+C 또는 창 닫기")

    step_count = 0

    # ---  영상 저장 설정 ---
    metadata = dict(title='Narrow Passage Simulation', artist='Geonwoo Kim', 
                   comment='8 agents passing through narrow corridor')
    writer = PillowWriter(fps=5)
    output_filename = "narrow_passage_simulation.gif"

    # fig 객체 가져오기
    fig = visualizer.fig if hasattr(visualizer, 'fig') else plt.gcf()

    try:
        with writer.saving(fig, output_filename, dpi=150):
            while True:
                # GBP 기반 경로 계획
                env.step_plan(iters=8)

                # 시각화 업데이트
                visualizer.update_visualization()

                # 프레임 저장
                writer.grab_frame()

                # 에이전트 이동
                env.step_move()

                if step_count % 10 == 0:
                    print(f"\n[Step {step_count}]")
                    
                    for i, agent in enumerate([agent0, agent1, agent2]):
                        state = agent.get_state()
                        if state[0] is not None:
                            x, y, vx, vy = state[0]
                            target = agent.get_target()
                            dist_to_goal = np.linalg.norm([x - target[0, 0], y - target[1, 0]])
                            print(f"    {agent.name}: pos=({x:.1f}, {y:.1f}), dist={dist_to_goal:.1f}")
                    
                    for i, agent in enumerate([agent3, agent4, agent5]):
                        state = agent.get_state()
                        if state[0] is not None:
                            x, y, vx, vy = state[0]
                            target = agent.get_target()
                            dist_to_goal = np.linalg.norm([x - target[0, 0], y - target[1, 0]])
                            print(f"    {agent.name}: pos=({x:.1f}, {y:.1f}), dist={dist_to_goal:.1f}")

                step_count += 1

                # 모든 에이전트가 목표에 도달했는지 확인
                all_reached = True
                for agent in agents:
                    state = agent.get_state()
                    if state[0] is not None: 
                        x, y, _, _ = state[0]
                        target = agent.get_target()
                        dist = np.linalg.norm([x - target[0, 0], y - target[1, 0]])
                        if dist > 10:  
                            all_reached = False
                            break
                
                if all_reached:
                    print("\n" + "=" * 60)
                    print("모든 에이전트가 목표에 도달했습니다!")
                    print(f"총 {step_count} 스텝 소요")
                    print(f"결과 저장: '{output_filename}'")
                    print("=" * 60)
                    
                    # 마지막 프레임 몇 개 더 저장 (정지 화면)
                    for _ in range(10):
                        writer.grab_frame()
                    
                    plt.pause(3)
                    break
                
                # 최대 스텝 제한 (무한 루프 방지)
                if step_count > 500:
                    print("\n 최대 스텝 도달 - 시뮬레이션 종료")
                    break

    except KeyboardInterrupt:
        print("\n시뮬레이션 중단됨 (사용자 입력)")

    finally:
        print("\n시뮬레이션 종료")
        visualizer.close()