from PIL import Image
from torchvision import transforms as T
import numpy as np
from PIL import Image

#aspect ratio constants
standing_aspect = 290/490 #0.591 ~ Images where the subject is wider than this get rejected
breadth_flank_aspect = 1/3.4 #0.29 ~ Images where the subject's hips & waist width aren't at least this portion of their flank the image gets rejected
vert_aspect = 21/9 #2.333 ~ Images where the top and bottom of subject's flank aren't algined to a bounding box of this ratio the image gets rejected 
hor_aspect = 10/16 #0.625 ~ Images where the left and right of a subject's hips / waist aren't aligned to a bounding box of this ratio the image gets rejected


def analyse_entry(index, mask, mask_w, mask_h, tensor):

    #Our current razor for image selection considers four factors:
    #Is the subject the sole human figure in the frame?
    #Can every keypoint of the body be recognised with confidence? (fullbody test)
    #Does the person have their left and right keypoints aligned? (vertical keypoints bounding box 21:9 or wider)
    #Does the person have their top and bottom keypoints aligned? (horizontal keypoints bounding box 10:16 or thinner)

    print("Processing row ", index, "...")

    results = [0, 0, 0, 0]
    
    results[0] = is_person_solo(tensor)
    
    if results[0] == 1:
        results[1] = is_person_whole(tensor)
    
    if results[1] == 1:
        results[2] = is_person_standing(tensor, mask, mask_w, mask_h)
        results[3] = is_person_facing_straight(tensor)
    
    return results

def is_person_solo(tensor):
    
    #Check number of subjects in image
    subjects = tensor['scores'].tolist()

    #Iterate backwards through detected subjects and weed out hallucinations
    for x in reversed(range(len(subjects))):
        if subjects[x] < 0.9:
            del subjects[x]

    num_subjects = len(subjects)

    if num_subjects == 1:
        return 1
    else: 
        return 0

def is_person_whole(tensor):

    keypoints = tensor['keypoints'][0]
    subject_keypoint_scores = tensor['keypoints_scores'][0]    
    subject_keypoint_list = keypoints.tolist()
    subject_keypoint_scores_list = subject_keypoint_scores.tolist()
    
    for x in reversed(range(len(subject_keypoint_list))):
        if subject_keypoint_scores_list[x] < 4:
            del subject_keypoint_list[x]
            del subject_keypoint_scores_list[x]
    
    #keypointrcnn_resnet50_fpn returns 17 keypoints
    if len(subject_keypoint_scores_list) == 17:
        return 1
    else: 
        return 0

def is_person_standing(tensor, mask, mask_w, mask_h):

    #This is the most problematic function in this set so it's quite verbose

    # Standing detection consists of a simple "razor" using aspect ratio checking and then measuring the uprightness of a person and neutrality of the photo
    
    #We'll scan each pixel in the image to make a bounding box around the non-transparent subject
    #Initialise leftmost as highest and rightmost as smallest so they can be brought down to their
    #largest and smallest points respectively
    leftmost_pixel = mask_w
    rightmost_pixel = 0
    topmost_pixel = mask_h
    bottommost_pixel = 0
    for x in range(mask_w):
        for y in range(mask_h):
            pixel = mask.getpixel((x, y))
            if pixel[3] == 255: # Check that the pixel's fully opaque
                if x < leftmost_pixel:
                    leftmost_pixel = x
                if x > rightmost_pixel:
                    rightmost_pixel = x
                if y < topmost_pixel:
                    topmost_pixel = y
                if y > bottommost_pixel:
                    bottommost_pixel = y

    # Get aspect ratio
    subject_width = rightmost_pixel - leftmost_pixel + 1
    subject_height = bottommost_pixel - topmost_pixel + 1
    subject_aspect = subject_width / subject_height 

    print("Subject aspect ratio is ", subject_aspect)
    print("Maximum Reference aspect for standing is ", standing_aspect)

    if subject_aspect > standing_aspect:
            return 0

    # Where aspect razor passed, check keypoint analysis
    keypoints = tensor['keypoints'][0]

    subject_keypoints = {
    "nose" : keypoints[0].cpu().detach().numpy(), 
    "left_eye" : keypoints[1].cpu().detach().numpy(), 
    "right_eye" : keypoints[2].cpu().detach().numpy(),
    "left_ear": keypoints[3].cpu().detach().numpy(), 
    "right_ear": keypoints[4].cpu().detach().numpy(), 
    "left_shoulder": keypoints[5].cpu().detach().numpy(), 
    "right_shoulder": keypoints[6].cpu().detach().numpy(), 
    "left_elbow": keypoints[7].cpu().detach().numpy(),
    "right_elbow": keypoints[8].cpu().detach().numpy(), 
    "left_wrist": keypoints[9].cpu().detach().numpy(), 
    "right_wrist": keypoints[10].cpu().detach().numpy(), 
    "left_hip": keypoints[11].cpu().detach().numpy(), 
    "right_hip": keypoints[12].cpu().detach().numpy(), 
    "left_knee": keypoints[13].cpu().detach().numpy(), 
    "right_knee": keypoints[14].cpu().detach().numpy(), 
    "left_ankle": keypoints[15].cpu().detach().numpy(), 
    "right_ankle": keypoints[16].cpu().detach().numpy()
    }

    #Subject keypoints can be any variety of uneven, so let's approximate their section y co-ord by splitting the diff
    shoulder_center = (subject_keypoints["left_shoulder"][1] + subject_keypoints['right_shoulder'][1]) / 2
    hip_center = (subject_keypoints["left_hip"][1] + subject_keypoints['right_hip'][1]) / 2
    knee_center = (subject_keypoints["left_knee"][1] + subject_keypoints['right_knee'][1]) / 2
    ankle_center = (subject_keypoints["left_ankle"][1] + subject_keypoints['right_ankle'][1]) / 2


    #Hair will throw off all head measurements, but we don't need to know about the position of the head.
    #Let's consider the height what constitues the length of the average person without their head or neck:
    
    head_neck_height = shoulder_center - topmost_pixel
    print("Height of subject head and neck = ", head_neck_height)
    adjusted_height = subject_height - head_neck_height
    print("Total height - ^ = ", adjusted_height)
    print("Original height = ", subject_height)

    #If a subject is sitting, kneeling, lounging, etc, the height composed by one section will be off by a lot


    #Now we can deduce each section's proportional length
    perc_torso = abs((hip_center - shoulder_center) / adjusted_height)
    perc_thigh = abs((knee_center - hip_center) / adjusted_height)
    perc_calf = abs((ankle_center - knee_center) / adjusted_height)

    #Let's check that the proportions are what we expect
    if perc_torso > 0.52 or perc_torso < 0.29:
        print("Subject's torso appears too long / short, composing ", perc_torso, " of their height." )
        return 0
    if perc_thigh < 0.21 or perc_thigh > 0.375:
        print("Subject's thighs appears too long / short, composing ", perc_thigh, " of their height." )
        return 0
    if perc_calf < 0.15 or perc_calf > 0.32:
        print("Subject's calves appears too long / short, composing ", perc_calf, " of their height." )
        return 0
    
    print("Subject appears to be standing.")
    print("Shoulder center = ", shoulder_center)
    print("Hip center = ", hip_center)
    print("Knee center = ", knee_center)
    print("Ankle center = ", ankle_center)
    print("Ratios = ", perc_torso, perc_thigh, perc_calf)

    return 1


def is_person_facing_straight(tensor):

    keypoints = tensor['keypoints'][0]

    subject_keypoints = {
    "nose" : keypoints[0].cpu().detach().numpy(), 
    "left_eye" : keypoints[1].cpu().detach().numpy(), 
    "right_eye" : keypoints[2].cpu().detach().numpy(),
    "left_ear": keypoints[3].cpu().detach().numpy(), 
    "right_ear": keypoints[4].cpu().detach().numpy(), 
    "left_shoulder": keypoints[5].cpu().detach().numpy(), 
    "right_shoulder": keypoints[6].cpu().detach().numpy(), 
    "left_elbow": keypoints[7].cpu().detach().numpy(),
    "right_elbow": keypoints[8].cpu().detach().numpy(), 
    "left_wrist": keypoints[9].cpu().detach().numpy(), 
    "right_wrist": keypoints[10].cpu().detach().numpy(), 
    "left_hip": keypoints[11].cpu().detach().numpy(), 
    "right_hip": keypoints[12].cpu().detach().numpy(), 
    "left_knee": keypoints[13].cpu().detach().numpy(), 
    "right_knee": keypoints[14].cpu().detach().numpy(), 
    "left_ankle": keypoints[15].cpu().detach().numpy(), 
    "right_ankle": keypoints[16].cpu().detach().numpy()
    }

    #Let's access the coordinates of each anchoring keypoint in the torso 
    left_shoulder_x, left_shoulder_y = subject_keypoints['left_shoulder'][0], subject_keypoints['left_shoulder'][1]
    right_shoulder_x, right_shoulder_y = subject_keypoints['right_shoulder'][0], subject_keypoints['right_shoulder'][1] 
    left_hip_x, left_hip_y = subject_keypoints['left_hip'][0], subject_keypoints['left_hip'][1]
    right_hip_x, right_hip_y = subject_keypoints['right_hip'][0], subject_keypoints['right_hip'][1]
    
    #Now let's draw bounding boxes around the shoulders, hips, and left and right sides of the body 
    shoulder_box_width, shoulder_box_height = (left_shoulder_x - right_shoulder_x), (right_shoulder_y - left_shoulder_y)
    hip_box_width, hip_box_height = (right_hip_x - left_hip_x), abs((right_hip_y - left_hip_y))
    flank_height = ((left_hip_y - left_shoulder_y) + (right_hip_y - right_shoulder_y)) / 2
    left_box_width, left_box_height = (left_shoulder_x - left_hip_x), abs((left_shoulder_y - left_hip_y))
    right_box_width, right_box_height = (right_shoulder_x - right_hip_x), abs((right_shoulder_y - right_hip_y))
    
    #Let's check that the subject's hips and shoulders are roughly level with each other by checking the bounding box's aspect ratio
    shoulder_aspect_ratio = abs(shoulder_box_width / shoulder_box_height)
    hip_aspect_ratio = abs(hip_box_width / hip_box_height)
    shoulder_flank_aspect_ratio = abs(shoulder_box_width / flank_height)
    hip_flank_aspect_ratio = abs(hip_box_width / flank_height)

    #Let's check that the subject's left and right sides are being help up straight
    left_aspect_ratio = left_box_width / left_box_height
    right_aspect_ratio = right_box_width / right_box_height

    if (shoulder_aspect_ratio > vert_aspect) and (hip_aspect_ratio > vert_aspect) and (left_aspect_ratio < hor_aspect) and (right_aspect_ratio < hor_aspect) and (shoulder_flank_aspect_ratio > breadth_flank_aspect) and (hip_flank_aspect_ratio > breadth_flank_aspect):
        print("Subject is standing straight")
        print("Shoulder Coords: ", left_shoulder_x, left_shoulder_y, " | ", right_shoulder_x, right_shoulder_y)
        print("Hip Coords: ", left_hip_x, left_hip_y, " | ", right_hip_x, right_hip_y)
        print("Shoulder Box = ", shoulder_box_width, " ", shoulder_box_height, "Hip Box = ", hip_box_width, " ", hip_box_height, "Knee Box = ")
        print("Left Box = ", left_box_width, " ", left_box_height, "Right Box = ", right_box_width, " ", right_box_height)
        print("Shoulder Width = ", shoulder_box_width, " Flank Height = ", flank_height)
        print("Shoulder Flank Aspect Ratio", shoulder_flank_aspect_ratio, "Hip Flank Aspect Ratio", hip_flank_aspect_ratio, "Expected Breadth Flank Aspect Ratio = ", breadth_flank_aspect)
        print("Shoulder, hip, left, right aspect ratios:", shoulder_aspect_ratio, hip_aspect_ratio, left_aspect_ratio, right_aspect_ratio)
        return 1
    else: 
        print("Subject in image not standing straight.")
        print("Shoulder Coords: ", left_shoulder_x, left_shoulder_y, " | ", right_shoulder_x, right_shoulder_y)
        print("Hip Coords: ", left_hip_x, left_hip_y, " | ", right_hip_x, right_hip_y)
        print("Shoulder Box = ", shoulder_box_width, " ", shoulder_box_height, "Hip Box = ", hip_box_width, " ", hip_box_height, "Knee Box = ")
        print("Left Box = ", left_box_width, " ", left_box_height, "Right Box = ", right_box_width, " ", right_box_height)
        print("Shoulder Width = ", shoulder_box_width, " Flank Height = ", flank_height)
        print("Shoulder Flank Aspect Ratio", shoulder_flank_aspect_ratio, "Hip Flank Aspect Ratio", hip_flank_aspect_ratio, "Expected Breadth Flank Aspect Ratio = ", breadth_flank_aspect)
        print("Shoulder, hip, left, right aspect ratios:", shoulder_aspect_ratio, hip_aspect_ratio, left_aspect_ratio, right_aspect_ratio)
        return 0

