import cv2

def detect_lines(image_path, output):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    lsd = cv2.createLineSegmentDetector(0)
    lines, _, _, _ = lsd.detect(gray)

    drawn_image = lsd.drawSegments(image, lines)
    cv2.imwrite(output, drawn_image)

if __name__ == "__main__":
    image_path = "/home/wenhuanyao/317VO/demo/demo_pic.png"
    output = "/home/wenhuanyao/317VO/results/demo_pic_lsd.png"
    detect_lines(image_path, output)
    print("finished")