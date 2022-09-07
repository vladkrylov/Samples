import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_ratio(contours, area_thr):
    areas = np.array([len(contour) for contour in contours])
    large_contour_ids = areas > area_thr

    ratio = areas[large_contour_ids].sum() / areas.sum()
    large_contours = [
        contour for i, contour in enumerate(contours)
        if large_contour_ids[i]
    ]

    return ratio, large_contours


def add_label(frame, text, position=(10, 30)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.7
    fontColor = (255,255,255)
    lineType = 2

    cv2.putText(
        frame,
        text, 
        position, 
        font, 
        fontScale,
        fontColor,
        lineType
    )


def plot_results(ratios, fps):
    mean_ratio = np.array(ratios).mean()

    plt.figure(figsize=(12, 8))
    time = np.arange(len(ratios)) / fps
    plt.plot(time, ratios)
    plt.plot([time.min(), time.max()], [mean_ratio] * 2, '--')
    plt.xlabel('Time, sec')
    plt.ylabel('Large fraction ratio')
    plt.title(f'Mean ratio = {mean_ratio:.2f}')
    plt.grid(True)
    plt.legend(['Instant ratio', 'Mean ratio'])
    plt.show()


def main():
    cap = cv2.VideoCapture('data/1.mp4')
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.mp4', fourcc, fps, (980, 720))

    hsv_min = (0, 0, 150)
    hsv_max = (255, 180, 255)
    ratios = []
    while True:
        _, frame = cap.read()
        if frame is None:
            break

        # cut sides
        frame = frame[:, 200:-100, :]

        # get foreground
        frame_blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(frame_blurred, cv2.COLOR_BGR2HSV)
        fgmask = cv2.inRange(hsv, hsv_min, hsv_max)
        fgmask = cv2.dilate(fgmask, None, iterations=1)

        # extract contours
        contours, _ = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # get large fraction ratio
        ratio, large_contours = get_ratio(contours, area_thr=30)
        ratios.append(ratio)

        # compose output frame
        cv2.drawContours(frame, contours, -1, (0, 0, 255), 2)

        large_contour_mask = np.zeros_like(frame)
        cv2.fillPoly(large_contour_mask, pts=large_contours, color=(-127, -127, 240))
        frame = cv2.addWeighted(frame, 1, large_contour_mask, -1, 0)

        add_label(frame, f'Large fraction ratio = {ratio:.2f}')

        out.write(frame)
        cv2.imshow('Frame', frame)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    plot_results(ratios, fps)

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
