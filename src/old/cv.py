import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import linear_sum_assignment


def get_triangles(file_path):
    image_raw = cv2.imread(file_path) #afbeelding inlezen
    if image_raw is None:
        return None, []

    scale = 800 / image_raw.shape[1] #afbeelding schalen naar 800 pixels
    img = cv2.resize(image_raw, (800, int(image_raw.shape[0] * scale)))
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #afbeelding omzetten naar grijswaarden
    blur = cv2.GaussianBlur(gray, (3, 3), 0) #Guassian blur om kleine blur te voorkomen maar (3, 3) relatief kleine blur

    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU) #zet grijswaarden om in zwart-witafbeelding en OpenCV kiest automatisch geschikte grenswaarde
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #OpenCV zoekt omtrekken witte objecten enkel buitenomtrek
    
    triangles = [] #lijst om gevonden driehoeken aan toe te voegen

    for cnt in contours: #loopt door alle gevonden contouren door om ruis te vermijden
        if cv2.contourArea(cnt) < 50:  #gevonden driehoeken kleiner dan 50 pixels genegeerd 
            continue
            
        peri = cv2.arcLength(cnt, True) #lengte contourrand berekend 
        approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
        
        if len(approx) == 3:  #enkel contouren met 3 hoeken geaccepteerd als driehoek
            M = cv2.moments(cnt)
            if M["m00"] == 0: continue #zoekt centrum driehoek
            cX = int(M["m10"] / M["m00"])  #centrum contour x
            cY = int(M["m01"] / M["m00"]) #centrum contour y

            points = approx.reshape(-1, 2) #hoekpunten van de driehoek opgehaald

            sorted_points = points[points[:, 1].argsort()] 
            p1, p2 = sorted_points[1], sorted_points[2]
            dx = p2[0] - p1[0] #vector van de basis
            dy = p2[1] - p1[1]
            
            hoek = math.degrees(math.atan2(dy, dx)) #hoek basislijn berekend

            if hoek > 90:
                hoek -= 180
            elif hoek < -90:
                hoek += 180
            
            triangles.append({'pos': (cX, cY),'hoek': hoek,'cnt': approx}) #driehoeken opgeslagen in de lijst met positie, hoek en benaderde contour
            
    return img, triangles


def angle_difference_deg(a, b):
    diff = a - b
    diff = (diff + 180) % 360 - 180
    return diff


def match_triangles_global(ref_triangles, new_triangles, max_cost=150, angle_weight=2.0):


    n_ref = len(ref_triangles) #counts how many reference triangles detected
    n_new = len(new_triangles) #counts how many new triangles detected

    if n_ref == 0 or n_new == 0: #list will be empty when it can't match the both
        return []

    cost_matrix = np.zeros((n_new, n_ref)) #via een cost matrix krijg je de grootst mogelijke kans dat ze matchen

    for i, new_tri in enumerate(new_triangles):
        for j, ref_tri in enumerate(ref_triangles):
            dx = new_tri['pos'][0] - ref_tri['pos'][0] # verplaatsing tussen de nieuwe en referentie driehoek
            dy = new_tri['pos'][1] - ref_tri['pos'][1]

            pos_dist = math.sqrt(dx**2 + dy**2)

            angle_diff = abs(angle_difference_deg(new_tri['hoek'], ref_tri['hoek']))
            cost = pos_dist + angle_weight * angle_diff
            cost_matrix[i, j] = cost

    new_indices, ref_indices = linear_sum_assignment(cost_matrix)
    matches = []

    for new_i, ref_j in zip(new_indices, ref_indices):
        cost = cost_matrix[new_i, ref_j]

        if cost <= max_cost:
            new_tri = new_triangles[new_i]
            ref_tri = ref_triangles[ref_j]

            dx = new_tri['pos'][0] - ref_tri['pos'][0]
            dy = new_tri['pos'][1] - ref_tri['pos'][1]
            dtheta = angle_difference_deg(new_tri['hoek'], ref_tri['hoek'])

            matches.append({
                "new_index": new_i,
                "ref_index": ref_j,
                "new_tri": new_tri,
                "ref_tri": ref_tri,
                "new_pos": new_tri["pos"],
                "ref_pos": ref_tri["pos"],
                "dx": dx,
                "dy": dy,
                "dtheta": dtheta,
                "cost": cost
            })

    return matches

ref_img, ref_triangles = get_triangles(r"./src/ref.png")  #dit is de referentie of beginwaarden
new_img, new_triangles = get_triangles(r"./src/new.png") #hier wordt nieuwe afbeelding ingelezen, maar dit wordt de output van Ross zijn gegenereerde afbeeldingen

print(f"Aantal referentie driehoeken gevonden: {len(ref_triangles)}")
print(f"Aantal nieuwe driehoeken gevonden: {len(new_triangles)}")  #aantal driehoeken in nieuwe

matches = match_triangles_global(
    ref_triangles,
    new_triangles,
    max_cost=150,
    angle_weight=2.0
)

print(f"Aantal matches gevonden: {len(matches)}")

for match in matches:
    new_i = match["new_index"]
    ref_i = match["ref_index"]

    new_tri = match["new_tri"]
    ref_tri = match["ref_tri"]

    verplaatsing_x = match["dx"]
    verplaatsing_y = match["dy"]
    verschil_hoek = match["dtheta"]
    cv2.drawContours(new_img, [new_tri['cnt']], -1,(0, 255, 0), 2)# contour van nieuwe driehoek
    cv2.putText(new_img, str(ref_i),(new_tri['pos'][0] - 10, new_tri['pos'][1] + 5), #matcht een nummer met elke afbeelding hier gaat het van rechtsonder naar linksboven
        cv2.FONT_HERSHEY_SIMPLEX, 0.4,(255, 0, 0),1)

    print(
        f"Ref driehoek {ref_i} -> Nieuwe driehoek {new_i}: "
        f"RefPos={ref_tri['pos']}, "
        f"NewPos={new_tri['pos']}, "
        f"Verplaatsing=({verplaatsing_x}, {verplaatsing_y}), "
        f"Hoekverschil={verschil_hoek:.2f}, "
        f"Cost={match['cost']:.2f}"
    )

matched_ref_indices = set(match["ref_index"] for match in matches)
all_ref_indices = set(range(len(ref_triangles)))

unmatched_ref_indices = all_ref_indices - matched_ref_indices

if unmatched_ref_indices:
    print("\nNiet-gematchte referentiedriehoeken:")
    for idx in sorted(unmatched_ref_indices):
        print(f"Ref driehoek {idx}, Pos={ref_triangles[idx]['pos']}")


plt.figure(figsize=(12, 8))
plt.imshow(cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB))
plt.title("Genummerde Driehoeken met Global Matching")
plt.axis('off')
plt.show()