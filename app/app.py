import numpy as np
import cv2
import copy
from jinja2 import Environment, FileSystemLoader
from sys import argv


class Note(object):
    pitches = ('A2', 'G2', 'F2', 'E2', 'D2', 'C2', 'B', 'A', 'G', 'F', 'E', 'D', 'C')

    def __init__(self, center):
        self.pitch = 'unknown'
        self.duration = 'quarter'
        self.center = center
        self.x = center[0]
        self.y = center[1]

    def __repr__(self):
        return '<Note {}, at loc {}>'.format(self.pitch, self.center)

    def location(self):
        return int(self.x), int(self.y)


class Symbol(object):

    def __init__(self, type, location):
        self.type = type
        self.location = location
        self.x = location[0]
        self.y = location[1]

    def __repr__(self):
        return '<Symbol {}, at {}>'.format(self.type, self.location)

    def __eq__(self, other):
        if other.x - 5 <= self.x <= other.x + 5:
            if other.y - 5 <= self.y <= other.y + 5:
                return True
        return False


def find_closest_note(notes, eight_note):
    x = eight_note.location[0]
    distances = [abs(note.x - x) for note in notes]
    return notes[distances.index(min(distances))]


def find_eights(notes, matches, quarter, staves):
    for eight in quarter:
        if eight not in matches.keys():
            continue
        for eight_note in matches[eight]:
            for note_group, staff in zip(notes, staves):
                min_y = staff[0]
                max_y = staff[-1]
                y = eight_note.location[1]
                if min_y <= y <= max_y:
                    n = find_closest_note(note_group, eight_note)
                    n.duration = 'eigth'
                    break


def find_pitch(note, staff):
    distances = [abs(note.y - y) for y in staff]
    return Note.pitches[distances.index(min(distances))]


def classify_notes(segregated_notes, staves):
    for notes, staff in zip(segregated_notes, staves):
        for note in notes:
            note.pitch = find_pitch(note, staff)


def get_matches_with_loc(img_rgb, img_gray, templates, threshold):
    height, width, nn = img_rgb.shape
    locations = {}
    for template_symbol in templates:
        locations[template_symbol] = []
        fname = 'templates/' + template_symbol + '.jpg'
        template = cv2.imread(fname, 0)
        w, h = template.shape[::-1]
        o = height / config[template_symbol][1]
        o2 = width / config[template_symbol][0]
        for i in np.arange(o - 0.1, o + 0.1, 0.1):
            for j in np.arange(o2 - 0.1, o2 + 0.2, 0.1):
                changed = cv2.resize(template, (0, 0), fx=i, fy=j)
                res = cv2.matchTemplate(img_gray, changed, cv2.TM_CCOEFF_NORMED)
                loc = np.where(res >= threshold)
                pts = list(zip(*loc[::-1]))
                for pt in pts:
                    x = pt[0]
                    y = pt[1]
                    pt2 = (pt[0] + int(w * i), pt[1] + int(h * j))
                    x2 = pt2[0]
                    y2 = pt2[1]
                    center = (int((x + x2) / 2), int((y + y2) / 2))
                    s = Symbol(template_symbol, center)
                    if s not in locations[template_symbol]:
                        locations[template_symbol].append(s)
                    cv2.rectangle(img_rgb, pt, pt2, (255, 255, 255), cv2.FILLED)
                    cv2.rectangle(img_gray, pt, pt2, (255, 255, 255), cv2.FILLED)
                    # cv2.circle(img_rgb, s.location, 1, (255, 0, 0), 3)
                    if template_symbol == 'viol-key' or template_symbol == 'viol-key.3' or template_symbol == 'bas-key':
                        img_rgb[:pt[1] + int(h * j), :pt[0] + int(w * i)] = (255, 255, 255)
                if pts:
                    break
            else:
                continue
            break
    return locations


config = {
    '4.4': (1920, 1002),
    '4.4.2': (898, 845),
    '3.4': (1128, 570),
    '2.4.2': (957, 612),
    '3.4.2': (1144, 538),
    '2.4': (748, 502),
    'b': (1128, 570),
    'b.2': (1128, 570),
    'pauza_cw': (1920, 962),
    'pauza_cw.2': (1128, 570),
    'hasz': (1218, 609),
    'hasz2': (1218, 609),
    'hasz3': (1920, 962),
    'post': (1920, 962),
    'bas-key': (1920, 962),
    'viol-key': (1920, 1002),
    'viol-key.2': (960, 698),
    'viol-key.3': (898, 845),
    'half': (1218, 609),
    'half.2': (1218, 609),
    'half.3': (898, 845),
    'half.4': (1920, 1002),
    'half.5': (1920, 1002),
    'half.6': (1128, 570),
    'half.7': (1128, 570),
    'half.8': (898, 845),
    'half.9': (1128, 570),
    'quater': (1218, 609),
    'quater.2': (960, 698),
    'quater.3': (758, 502),
    'quater.4': (905, 534),
    'quater.5': (957, 612),
    'whole': (1920, 1002)
}

images = ['low.jpg', 'kurki-trzy.jpg', 'stoo.jpg', 'dudka.jpg', 'Happy.jpg', 'oda.jpg', 'kotek.jpg', 'Lulajże.jpg',
          'aaakot.jpg', 'kurki3.jpg', 'niewiem.jpg']
metrum = ('4.4', '3.4', '2.4.2', '3.4.2', '2.4', '4.4.2')
pause = ('pauza_cw', 'pauza_cw.2')
hasz = ('hasz', 'hasz2', 'hasz3')
b = ('b', 'b.2')
post = (('post'),)
bas_key = (('bas-key'),)
viol_key = ('viol-key', 'viol-key.3')
half = ('half.4', 'half.5', 'half.7')
quater = ('quater', 'quater.2')
whole = (('whole'),)
symbols = {metrum: 0.655, pause: 0.7, hasz: 0.8, b: 0.9, post: 0.739, bas_key: 0.7, viol_key: 0.55}
symbols2 = {half: 0.66, quater: 0.63, whole: 0.65}

image = argv[1]
img = cv2.imread(image)
if img is None:
    exit(1)
cimg = copy.copy(img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
canny = cv2.Canny(gray, 50, 150, apertureSize=3)
lines = cv2.HoughLines(canny, 1, np.pi / 180, int(0.4 * img.shape[1]))
line_coords = []
for line in lines:
    for r, theta in line:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * r
        y0 = b * r
        x1 = int(x0 + img.shape[1] * (-b))
        y1 = int(y0 + img.shape[1] * a)
        x2 = int(x0 - img.shape[1] * (-b))
        y2 = int(y0 - img.shape[1] * a)
        cv2.line(cimg, (x1, y1), (x2, y2), (0, 255, 0), 1)
        line_coords.append([(x1, y1), (x2, y2)])


line_coords[:] = sorted(line_coords, key=lambda x: x[0][1])[::2]
diff = line_coords[1][0][1] - line_coords[0][0][1]
diff_ratio = diff * 100 / img.shape[0]

print('name: {}, diff: {}, diff ratio {}, numlines: {}'.format(image, diff, diff_ratio, len(line_coords)))

new_list = [[copy.deepcopy(line_coords[i + j * 5]) for i in range(5)] for j in range(int(len(line_coords) / 5))]

x1 = line_coords[0][0][0]
x2 = line_coords[0][1][0]

for elem in new_list:
    y1 = elem[0][1][1]
    y2 = elem[4][1][1]
    elem.insert(0, [(x1, y1 - diff), (x2, y1 - diff)])
    elem.append([(x1, y2 + diff), (x2, y2 + diff)])

staves = [[float(w[0][1] + 1) for w in elem] for elem in new_list]
middles = [[(line + next_line) / 2 for line, next_line in zip(staff, staff[1:])] for staff in staves]
for i, mids in enumerate(middles):
    for j, mid in enumerate(mids):
        staves[i].insert(j, mid)

for i in range(len(staves)):
    staves[i].sort()

matches = [get_matches_with_loc(img, gray, symbol, threshold) for symbol, threshold in
           zip(symbols.keys(), symbols.values())]

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
m = {}
for match in matches:
    for key, value in zip(match.keys(), match.values()):
        if value:
            m[key] = value
matches = m

_, w, h = img.shape[::-1]
img[:int(line_coords[0][0][1] - diff - h), :] = (255, 255, 255)
img[int(line_coords[-1][-1][1]) + diff * 2:, :] = (255, 255, 255)
gray[:int(line_coords[0][0][1] - diff - h), :] = 255
gray[int(line_coords[-1][-1][1]) + diff * 2:, :] = 255
img[:, int(img.shape[1] * 0.95)] = (255, 255, 255)
gray[:, int(img.shape[1] * 0.95)] = 255
for staff, next_staff in zip(staves, staves[1:]):
    last = int(staff[-1]) + int(diff / 2) + 1
    first = int(next_staff[0]) - int(diff / 2) - 1
    img[last:first, :] = (255, 255, 255)
    gray[last:first, :] = 255

binarized = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 35, -10)

horizontal = copy.deepcopy(binarized)
horizontal_size = int(horizontal.shape[0] / 30.0)
horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
horizontal = cv2.erode(horizontal, horizontalStructure, (-1, -1))
horizontal = cv2.dilate(horizontal, horizontalStructure, (-1, -1))
horizontal = ~horizontal


vertical = copy.deepcopy(binarized)
vertical_size = int(vertical.shape[1] / 30.0)
verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))
vertical = cv2.erode(vertical, verticalStructure, (-1, -1))
vertical = cv2.dilate(vertical, verticalStructure, (-1, -1))
vertical = ~vertical


no_staff = binarized ^ horizontal
no_staff = cv2.dilate(no_staff, (2, 1))

no_staff = no_staff - ~vertical


reconnected = cv2.erode(no_staff, (6, 1), iterations=3)

_, im = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
im = cv2.dilate(im, np.ones((3, 3)), iterations=2)
im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

im2, contours, hierarchy = cv2.findContours(im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
avg_size = sum(cv2.contourArea(c) for c in contours[1:]) / len(contours[1:])

params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea = avg_size * diff_ratio * 0.29
params.maxArea = diff ** 2
params.filterByConvexity = False

detector = cv2.SimpleBlobDetector_create(params)
notes = [Note(p.pt) for p in detector.detect(im)]
for note in notes:
    cv2.circle(img, (int(note.x) - 1, int(note.y) + 1), int(diff * 1.2), (255, 255, 255), cv2.FILLED)
    cv2.circle(cimg, (int(note.x) - 1, int(note.y) + 1), 1, (255, 0, 0), 6)


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


matches2 = [get_matches_with_loc(img, gray, symbol, threshold) for symbol, threshold in
            zip(symbols2.keys(), symbols2.values())]

m = {}
for match in matches2:
    for key, value in zip(match.keys(), match.values()):
        if value:
            matches[key] = value
            m[key] = value
matches2 = m

for key in matches.keys():
    for symbol in matches[key]:
        loc = symbol.location
        topleft = (int(loc[0]) - int(diff / 2), int(loc[1]) - int(diff / 2))
        bottomright = (int(loc[0]) + int(diff / 2), int(loc[1]) + int(diff / 2))
        cv2.rectangle(cimg, topleft, bottomright, (0, 0, 255), 2)



for h in half:
    if h not in matches.keys():
        continue
    for half_note in matches[h]:
        n = Note(half_note.location)
        n.duration = 'half'
        notes.append(n)

for w in whole:
    if w not in matches.keys():
        continue
    for whole_note in matches[w]:
        n = Note(whole_note.location)
        n.duration = 'whole'
        notes.append(n)


segregated_notes = []
for staff in staves:
    this_staff = []
    for note in notes:
        if staff[0] - 2 <= note.y <= staff[-1] + 2:
            this_staff.append(note)
    segregated_notes.append(sorted(this_staff, key=lambda note: note.x))

notes = []
for notes_in_staff in segregated_notes:
    for note in notes_in_staff:
        notes.append(note)

classify_notes(segregated_notes, staves)
find_eights(segregated_notes, matches, quater, staves)

for notes_in_staff in segregated_notes:
   for note in notes_in_staff:
   	cv2.putText(cimg, note.pitch, (int(note.x) + 10, int(note.y) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)

cv2.imwrite(argv[1][:-4] + '-result.jpg', cimg)
output = Environment(loader=FileSystemLoader(searchpath=".")).get_template('song.xml').render(notes=notes)
with open(argv[1][:-4] + '.xml', 'w') as f:
    f.write(output)
