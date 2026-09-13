import os
import sys
import cv2
import numpy as np
import time
import threading
import weakref
from functools import lru_cache
from collections import OrderedDict
sys.path.append(os.path.abspath('/'))
from utils import load_toml_as_dict, config_bool

last_debug_print_time = 0.0
should_print_debug_info = False

orig_screen_width, orig_screen_height = 1920, 1080

states_path = r"./images/states/"

star_drops_path = r"./images/star_drop_types/"
images_with_star_drop = []
for file in os.listdir(star_drops_path):
    if "star_drop" in file:
        images_with_star_drop.append(file)

end_results_path = r"./images/end_results/"

region_data = load_toml_as_dict("./cfg/lobby_config.toml")['template_matching']
match_result_crop_region = region_data['match_result']
STATE_DETECTION_CONFIDENCE = float(
    load_toml_as_dict("cfg/bot_config.toml").get("state_detection_confidence", 0.75)
)
STATE_FINDER_DEBUG = config_bool(
    load_toml_as_dict("cfg/debug_settings.toml").get('state_finder_debug'), False
)

# Per worker, cache numeric scores only for the exact same frame object. This
# avoids copying identical crops once per template and never reuses stale data.
_match_cache = threading.local()


@lru_cache(maxsize=128)
def _scaled_region(current_width, current_height, region):
    orig_x, orig_y, orig_width, orig_height = region
    width_ratio = current_width / orig_screen_width
    height_ratio = current_height / orig_screen_height
    return (
        int(orig_x * width_ratio), int(orig_y * height_ratio),
        int(orig_width * width_ratio), int(orig_height * height_ratio),
    )


def _match_result_buffer(height, width):
    """Return a reusable per-thread OpenCV template-score matrix."""
    buffers = getattr(_match_cache, 'result_buffers', None)
    if buffers is None:
        buffers = {}
        _match_cache.result_buffers = buffers
    key = (height, width)
    result = buffers.get(key)
    if result is None:
        result = np.empty(key, dtype=np.float32)
        buffers[key] = result
    return result


def _prepare_frame_cache(frame):
    frame_ref = getattr(_match_cache, 'frame_ref', None)
    cached_frame = frame_ref() if frame_ref is not None else None
    if cached_frame is not frame:
        _match_cache.frame_ref = weakref.ref(frame)
        _match_cache.crops = {}
        _match_cache.scores = {}


def _frame_crop(frame, key, x, y, width, height):
    """Reuse an exact region view across templates evaluated on one frame."""
    _prepare_frame_cache(frame)
    crops = _match_cache.crops
    crop = crops.get(key)
    if crop is None:
        crop = frame[y:y + height, x:x + width]
        crops[key] = crop
    return crop


def _template_score(frame, crop, template, key):
    _prepare_frame_cache(frame)
    cache = _match_cache.scores
    if key in cache:
        return cache[key]
    history = getattr(_match_cache, 'history', None)
    if history is None:
        history = _match_cache.history = OrderedDict()
        _match_cache.history_bytes = 0
    # Production keys identify a template followed by its region. Share one
    # pixel snapshot across all templates in that region instead of copying
    # the end-result banner eight times.
    region_key = key[1:] if isinstance(key, tuple) and len(key) == 4 else key
    previous = history.get(region_key)
    if previous is not None:
        old_crop, scores = previous
        if old_crop.shape == crop.shape and np.array_equal(old_crop, crop):
            history.move_to_end(region_key)
            saved = scores.get(key)
            if saved is not None and saved[0] is template:
                cache[key] = saved[1]
                return saved[1]
        else:
            _match_cache.history_bytes -= old_crop.nbytes
            del history[region_key]
            previous = None
    result_height = crop.shape[0] - template.shape[0] + 1
    result_width = crop.shape[1] - template.shape[1] + 1
    result = cv2.matchTemplate(
        crop, template, cv2.TM_CCOEFF_NORMED,
        result=_match_result_buffer(result_height, result_width),
    )
    score = cv2.minMaxLoc(result)[1]
    cache[key] = score
    if previous is not None:
        previous[1][key] = (template, score)
        return score
    # Bound retained pixels independently of capture resolution. Reuse only
    # byte-identical regions; changed pixels always take the original matcher.
    budget = 8 * 1024 * 1024
    if crop.nbytes <= budget:
        while history and (_match_cache.history_bytes + crop.nbytes > budget
                           or len(history) >= 64):
            _, entry = history.popitem(last=False)
            _match_cache.history_bytes -= entry[0].nbytes
        history[region_key] = (crop.copy(), {key: (template, score)})
        _match_cache.history_bytes += crop.nbytes
    return score




def is_template_in_region(image, template_path, region, threshold=None):
    if threshold is None:
        threshold = STATE_DETECTION_CONFIDENCE
    current_height, current_width = image.shape[:2]
    region_key = tuple(region)
    new_x, new_y, new_width, new_height = _scaled_region(
        current_width, current_height, region_key
    )
    cropped_image = _frame_crop(
        image,
        (current_width, current_height, region_key),
        new_x, new_y, new_width, new_height,
    )
    loaded_template = load_template(template_path, current_width, current_height)
    if loaded_template is None or cropped_image.size == 0:
        return False
    if (
        cropped_image.shape[0] < loaded_template.shape[0]
        or cropped_image.shape[1] < loaded_template.shape[1]
    ):
        return False

    try:
        max_val = _template_score(
            image, cropped_image, loaded_template,
            (template_path, current_width, current_height, region_key)
        )
    except cv2.error as e:
        if should_print_debug_info:
            print(f"Template matching failed for {template_path}: {e}")
        return False

    if should_print_debug_info:
        print(f"Template matching for {template_path} in region {region} yielded max_val: {max_val}")
    return max_val > threshold


@lru_cache(maxsize=128)
def load_template(image_path, width, height):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load template: {image_path}")
        return None
    orig_height, orig_width = image.shape[:2]
    current_width_ratio, current_height_ratio = width / orig_screen_width, height / orig_screen_height
    resized_image = cv2.resize(image, (int(orig_width * current_width_ratio), int(orig_height * current_height_ratio)))
    resized_colored_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    return resized_colored_image

SHOWDOWN_PLACE_THRESHOLD = 0.9
_CHECK_SHOWDOWN_RESULTS = True
_CHECK_STANDARD_RESULTS = True
showdown_place_templates = {
    0: ["1st.png"],
    1: ["2nd.png"],
    2: ["3rd.png", "3rd_alt.png"],
    3: ["4th.png"]
}


def configure_game_result_detection(gamemodes):
    """Skip impossible end-screen templates for a known playstyle mode."""
    global _CHECK_SHOWDOWN_RESULTS, _CHECK_STANDARD_RESULTS
    if isinstance(gamemodes, str):
        modes = (gamemodes.lower(),)
    else:
        modes = tuple(str(mode).lower() for mode in (gamemodes or ()))
    if not modes or "all" in modes:
        _CHECK_SHOWDOWN_RESULTS = True
        _CHECK_STANDARD_RESULTS = True
        return
    has_showdown = any("showdown" in mode for mode in modes)
    has_standard = any(
        "3v3" in mode or "5v5" in mode for mode in modes
    )
    if not has_showdown and not has_standard:
        _CHECK_SHOWDOWN_RESULTS = True
        _CHECK_STANDARD_RESULTS = True
        return
    _CHECK_SHOWDOWN_RESULTS = has_showdown
    _CHECK_STANDARD_RESULTS = has_standard

def find_game_result(screenshot):
    if _CHECK_SHOWDOWN_RESULTS:
        for place, template_files in showdown_place_templates.items():
            for template_file in template_files:
                if is_template_in_region(
                        screenshot,
                        end_results_path + template_file,
                        match_result_crop_region,
                        threshold=SHOWDOWN_PLACE_THRESHOLD
                ):
                    return f"trio_showdown_{place}"
    if not _CHECK_STANDARD_RESULTS:
        return False
    is_victory = is_template_in_region(screenshot, end_results_path + 'victory.png', match_result_crop_region)
    if is_victory:
        return "victory"

    is_defeat = is_template_in_region(screenshot, end_results_path + 'defeat.png', match_result_crop_region)
    if is_defeat:
        return "defeat"

    is_draw = is_template_in_region(screenshot, end_results_path + 'draw.png', match_result_crop_region)
    if is_draw:
        return "draw"
    return False


def get_in_game_state(image):
    global last_debug_print_time, should_print_debug_info
    current_time = time.time()
    should_print_debug_info = STATE_FINDER_DEBUG and (current_time - last_debug_print_time >= 1.0)
    if should_print_debug_info:
        last_debug_print_time = current_time

    try:
        if should_print_debug_info: print("Checking for match result...")
        game_result = is_in_end_of_a_match(image)
        if game_result: return f"end_{game_result}"
        if should_print_debug_info: print("Checking for lobby...")
        if is_in_lobby(image): return "lobby"
        if should_print_debug_info: print("Checking for match making...")
        if is_in_match_making(image): return "match_making"
        if should_print_debug_info: print("Checking for brawler selection...")
        if is_in_brawler_selection(image): return "brawler_selection"
        if should_print_debug_info: print("Checking for shop")
        if is_in_shop(image): return "shop"
        if should_print_debug_info: print("Checking for offer popup...")
        if is_in_offer_popup(image): return "popup"
        if should_print_debug_info: print("Checking for brawl pass or star road (shop state)...")
        if is_in_brawl_pass(image) or is_in_star_road(image): return "shop"
        if should_print_debug_info: print("Checking for prestige milestone...")
        if is_in_prestige_milestone(image): return "prestige_milestone"
        if should_print_debug_info: print("Checking for star drop...")
        star_drop_type = is_in_star_drop(image)
        if star_drop_type:
            return f"star_drop_{star_drop_type}"
        if should_print_debug_info: print("Checking for trophy reward...")
        if is_in_trophy_reward(image):
            return "trophy_reward"

        return "match"
    finally:
        should_print_debug_info = False


def is_in_shop(image) -> bool:
    return is_template_in_region(image, states_path + 'powerpoint.png', region_data["powerpoint"])


def is_in_brawler_selection(image) -> bool:
    return is_template_in_region(image, states_path + 'brawler_menu_heart.png', region_data["brawler_menu_heart"]) or is_template_in_region(image, states_path + 'brawler_menu_search.png', region_data["brawler_menu_search"])


def is_in_offer_popup(image) -> bool:
    return is_template_in_region(image, states_path + 'close_popup.png', region_data["close_popup"])


def is_in_lobby(image) -> bool:
    return is_template_in_region(image, states_path + 'lobby_menu.png', region_data["lobby_menu"])


def is_in_end_of_a_match(image):
    return find_game_result(image)


def is_in_trophy_reward(image):
    return is_template_in_region(image, states_path + 'trophies_screen.png', region_data["trophies_screen"])


def is_in_brawl_pass(image):
    return is_template_in_region(image, states_path + 'brawl_pass_house.png', region_data['brawl_pass_house'])


def is_in_star_road(image):
    return is_template_in_region(image, states_path + "go_back_arrow.png", region_data['go_back_arrow'])


def is_in_match_making(image):
    return is_template_in_region(image, states_path + "exit_match_making.png", region_data['exit_match_making'])


def is_in_prestige_milestone(image):
    return (
        is_template_in_region(
            image, states_path + "prestige_continue.png",
            region_data['prestige_continue']
        )
        or find_lower_right_green_action(image) is not None
    )


def find_lower_right_green_action(image):
    """Locate a wide green post-match action button such as LET'S GO.

    The name is kept for compatibility, although current prestige layouts can
    place the button on either side of the lower half of the screen.
    """
    height, width = image.shape[:2]
    # Prestige variants have moved this control between the lower-left,
    # centre, and lower-right. Restricting the search to the bottom quarter
    # missed the newer LET'S GO layout. The button is still a large, bright,
    # horizontal UI element, which separates it from map tiles and icons.
    x1, y1 = int(width * 0.04), int(height * 0.52)
    crop = image[y1:int(height * 0.98), x1:int(width * 0.96)]
    if crop.size == 0:
        return None
    lower = np.array((35, 90, 105), dtype=np.uint8)
    upper = np.array((100, 255, 255), dtype=np.uint8)
    # scrcpy versions have returned both RGB and BGR arrays. Accept either
    # ordering so a backend update cannot silently disable this button.
    rgb_mask = cv2.inRange(cv2.cvtColor(crop, cv2.COLOR_RGB2HSV), lower, upper)
    bgr_mask = cv2.inRange(cv2.cvtColor(crop, cv2.COLOR_BGR2HSV), lower, upper)
    mask = cv2.bitwise_or(rgb_mask, bgr_mask)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(3, width // 320), max(3, height // 270))
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    candidates = []
    minimum_area = width * height * 0.002
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < minimum_area:
            continue
        local_x, local_y, button_width, button_height = cv2.boundingRect(contour)
        if button_height < height * 0.035 or button_width < width * 0.14:
            continue
        if button_width / max(button_height, 1) < 2.0:
            continue
        fill_ratio = area / max(button_width * button_height, 1)
        if fill_ratio < 0.42:
            continue
        candidates.append((
            area,
            x1 + local_x + button_width * 0.5,
            y1 + local_y + button_height * 0.5,
        ))
    if not candidates:
        return None
    _, center_x, center_y = max(candidates, key=lambda item: item[0])
    return int(center_x), int(center_y)


def is_in_star_drop(image):
    for image_filename in images_with_star_drop:
        if is_template_in_region(image, star_drops_path + image_filename, region_data['star_drop']):
            if "angelic" in image_filename.lower(): return "angelic"
            if "demonic" in image_filename.lower(): return "demonic"
            if "starr_nova" in image_filename.lower(): return "starr_nova"
            return "regular"
    return False


def is_underdog(image):
    return is_template_in_region(image, end_results_path + "underdog.png", region_data['underdog'])


def get_state(screenshot, previous_state=None):
    # Prioritize checks that can actually follow the established state. The
    # caller still requests periodic full scans for unexpected transitions.
    if previous_state == "match":
        game_result = is_in_end_of_a_match(screenshot)
        if game_result:
            state = f"end_{game_result}"
        elif is_in_lobby(screenshot):
            # Check lobby before generic green reward buttons: lobby Play is
            # also green and occupies the lower-right part of the screen.
            state = "lobby"
        else:
            # Reward templates contain bright shapes also seen in attacks,
            # supers and map decorations. They cannot legitimately replace a
            # live match before an end/lobby transition has been confirmed.
            state = "match"
    elif previous_state == "lobby" and is_in_lobby(screenshot):
        state = "lobby"
    elif previous_state == "match_making" and is_in_match_making(screenshot):
        state = "match_making"
    elif previous_state == "match_making":
        # Prestige/reward overlays cannot legitimately appear while the
        # matchmaking screen is still the established state. If the exit
        # template vanished, let the entity probe establish the match or let
        # the next full scan identify the actual post-match menu.
        state = "lobby" if is_in_lobby(screenshot) else "match"
    elif previous_state == "brawler_selection" and is_in_brawler_selection(screenshot):
        state = "brawler_selection"
    else:
        state = get_in_game_state(screenshot)
    if STATE_FINDER_DEBUG:
        cv2.imwrite(
            f"./debug_frames/state_screenshot_{state}_{len(os.listdir('./debug_frames'))}.png",
            cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB),
        )
    return state
