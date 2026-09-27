// Silent Running: ESP32-CAM (AI-Thinker, on the ESP32-CAM-MB USB board) sends its frames down the USB cable instead of WiFi.
//
// Why: over WiFi the board's PCB antenna sits against the wearer's head, the link's capacity drops below what the stream
// needs, TCP stalls on the lost packets and the picture freezes for seconds. A cable has no radio to shadow; WiFi stays off.
// Host side: silent_running/sources.py SerialSource (--source serial or serial:/dev/cu.usbserial-XXXX).
//
// Protocol, board -> host: messages "SRF" + type (1 byte) + length (uint32 little-endian) + a check byte of the length
// (XOR of the 4 length bytes and 0xA5: a damaged header is rejected at once) + the payload's CRC-32 (uint32 LE, zlib's) +
// payload. Host side: sources.serial_header.
//   type 'J': one JPEG frame (always the newest the sensor has: CAMERA_GRAB_LATEST).
//   type 'T': text: "ready" after boot, "ok <command>[ <result>]" / "err <command>" for every command, "err ..." on a
//   capture failure.
//   "ok" comes only after the frames buffered before the change are discarded: every frame after it has the new settings.
// Host -> board: one command per line.
//   set framesize <n> | set quality <n>
//   win <sx> <sy> <ex> <ey> <offx> <offy> <tx> <ty> <ox> <oy> <scale> <binning>   the sensor's set_res_raw, as /resolution
//     (what the numbers mean depends on the sensor: this board's is an OV3660, see sources.ov3660_window)
//   sensor   answers "ok sensor <esp_camera PID, hex>" (3660 for an OV3660)
// Flash: Arduino IDE or arduino-cli, board "AI Thinker ESP32-CAM" (esp32:esp32:esp32cam). Opening the port on macOS resets
// the board once (DTR/RTS are wired to EN/IO0); the host then waits for "ready" and keeps both released so it runs.

#include "esp_camera.h"
#include "esp_log.h"
#include "esp_rom_crc.h"

#define FB_COUNT 2
// CH340 on the ESP32-CAM-MB: 1.5 Mbaud = ~150 KB/s, 30 fps for frames up to 5 KB. Measured on the board (ESP32-CAM-MB, macOS): 2 Mbaud lost a byte in 60% of frames, 1.5 Mbaud 2%, 1 Mbaud 1%.
#define BAUD 1500000  // the host's ?baud= must match

// AI-Thinker ESP32-CAM pins
#define PWDN_GPIO_NUM 32
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM 0
#define SIOD_GPIO_NUM 26
#define SIOC_GPIO_NUM 27
#define Y9_GPIO_NUM 35
#define Y8_GPIO_NUM 34
#define Y7_GPIO_NUM 39
#define Y6_GPIO_NUM 36
#define Y5_GPIO_NUM 21
#define Y4_GPIO_NUM 19
#define Y3_GPIO_NUM 18
#define Y2_GPIO_NUM 5
#define VSYNC_GPIO_NUM 25
#define HREF_GPIO_NUM 23
#define PCLK_GPIO_NUM 22

static void send(char type, const uint8_t *data, uint32_t len) {
  uint8_t check = (uint8_t)(len ^ (len >> 8) ^ (len >> 16) ^ (len >> 24) ^ 0xA5);
  uint32_t crc = esp_rom_crc32_le(0, data, len);  // == zlib.crc32(data)
  uint8_t hdr[13] = {'S', 'R', 'F', (uint8_t)type, (uint8_t)len, (uint8_t)(len >> 8), (uint8_t)(len >> 16), (uint8_t)(len >> 24), check,
                     (uint8_t)crc, (uint8_t)(crc >> 8), (uint8_t)(crc >> 16), (uint8_t)(crc >> 24)};
  Serial.write(hdr, sizeof(hdr));
  Serial.write(data, len);
}

static void reply(const String &text) {
  send('T', (const uint8_t *)text.c_str(), text.length());
}

void setup() {
  Serial.setTxBufferSize(16384);  // before begin(): a frame streams out while the camera fills the next buffer
  Serial.begin(BAUD);
  Serial.setDebugOutput(false);         // the driver's own prints (e.g. "FB-OVF") would land inside the frame stream
  esp_log_level_set("*", ESP_LOG_NONE);
  camera_config_t config = {};
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size = FRAMESIZE_UXGA;  // sizes the JPEG buffers (w*h/5) for any frame size; HVGA is set below
  config.jpeg_quality = 16;
  config.fb_count = FB_COUNT;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.grab_mode = CAMERA_GRAB_LATEST;
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    reply("err camera init failed 0x" + String(err, HEX));
    delay(2000);
    ESP.restart();
  }
  sensor_t *s = esp_camera_sensor_get();
  s->set_framesize(s, FRAMESIZE_HVGA);  // the host sets the frame size and quality it wants on every open
  reply("ready");
}

static void handle(String line) {
  line.trim();
  if (!line.length()) return;
  sensor_t *s = esp_camera_sensor_get();
  int v[12], r = -1;
  if (line.startsWith("set framesize ")) {
    r = s->set_framesize(s, (framesize_t)line.substring(14).toInt());
  } else if (line.startsWith("set quality ")) {
    r = s->set_quality(s, line.substring(12).toInt());
  } else if (line.startsWith("win ") && sscanf(line.c_str() + 4, "%d %d %d %d %d %d %d %d %d %d %d %d", &v[0], &v[1], &v[2], &v[3], &v[4], &v[5],
                                                &v[6], &v[7], &v[8], &v[9], &v[10], &v[11]) == 12) {
    r = s->set_res_raw(s, v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10] != 0, v[11] != 0);
  } else if (line == "sensor") {
    reply("ok sensor " + String(s->id.PID, HEX));
    return;
  }
  for (int i = 0; r == 0 && i < FB_COUNT; i++) {  // frames captured before the change: never sent
    camera_fb_t *fb = esp_camera_fb_get();
    if (fb) esp_camera_fb_return(fb);
  }
  reply((r == 0 ? "ok " : "err ") + line);
}

static String cmd;

void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') {
      handle(cmd);
      cmd = "";
    } else if (cmd.length() < 96) {
      cmd += c;
    }
  }
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    reply("err frame capture failed");
    return;
  }
  send('J', fb->buf, fb->len);
  esp_camera_fb_return(fb);
}
