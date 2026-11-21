import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:video_player/video_player.dart';
import 'package:flutter_tts/flutter_tts.dart';

const backendUrl = String.fromEnvironment(
  'BACKEND_URL',
  defaultValue: 'http://10.0.2.2:8000',
);

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final cameras = await availableCameras();
  runApp(SignLanguageApp(cameras: cameras));
}

class SignLanguageApp extends StatelessWidget {
  const SignLanguageApp({super.key, required this.cameras});
  final List<CameraDescription> cameras;

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'ภาษามือไทย',
      theme: ThemeData(
        useMaterial3: true,
        fontFamily: 'sans',
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xff087f8c),
          brightness: Brightness.light,
            backgroundColor: Colors.white,
            body: SafeArea(
              child: LayoutBuilder(
                builder: (context, constraints) => SingleChildScrollView(
                  padding: const EdgeInsets.fromLTRB(20, 34, 20, 24),
                  child: ConstrainedBox(
                    constraints: BoxConstraints(minHeight: constraints.maxHeight - 58),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.stretch,
                      children: [
                        _buildInstructionBox(),
                        const SizedBox(height: 20),
                        _buildWrittenBox(),
                        const SizedBox(height: 170),
                        _buildBottomControls(camera),
                        if (_video != null) ...[
                          const SizedBox(height: 18),
                          AspectRatio(aspectRatio: _video!.value.aspectRatio, child: VideoPlayer(_video!)),
                        ],
                      ],
                    ),
                  ),
                ),
              ),
            ),
    int _videoCount = 0;
  final FlutterTts _tts = FlutterTts();

        Widget _buildInstructionBox() => Container(
              height: 220,
              alignment: Alignment.center,
              padding: const EdgeInsets.symmetric(horizontal: 28, vertical: 20),
              decoration: BoxDecoration(border: Border.all(color: Colors.black, width: 2)),
              child: const Text(
                'หน้าจอแสดงผล\n\nตามกล้องและสามารถตรวจจับมือได้\nหรือกดปุ่มสลับ\n(มุมซ้ายล่างจะแสดงเป็น วิดีโอโมเดลแทน)\nและจะแสดงได้มากกว่าหนึ่งรูปแบบ\nแบ่งได้ตามความชัดที่พอดี',
                textAlign: TextAlign.left,
                style: TextStyle(fontSize: 15, height: 1.35, color: Colors.black),
              ),
            );

  @override
          return SizedBox(
            width: 1,
            height: 1,
            child: camera?.value.isInitialized == true ? CameraPreview(camera!) : const SizedBox.shrink(),
          );
        }

        Widget _buildBottomControls(CameraController? camera) => Column(
              children: [
                Row(
                  crossAxisAlignment: CrossAxisAlignment.end,
                  children: [
                    Expanded(child: _outlineButton('มุมสลับ', Icons.flip_camera_ios_outlined, () => setState(() => _wordMode = !_wordMode))),
                    const SizedBox(width: 26),
                    Expanded(child: _outlineButton('ลิ้นคำที่มี', Icons.menu_book_outlined, _playWordVideo)),
                  ],
                ),
                const SizedBox(height: 20),
                Row(
                  children: [
                    Expanded(child: _outlineButton('ตรวจจับมือแบบคำ\nเปิดหรือปิด ตามชอบ', Icons.pan_tool_alt_outlined, () => setState(() => _characterEnabled = !_characterEnabled))),
                    const SizedBox(width: 12),
                    Expanded(child: _outlineButton('ตรวจจับมือแบบท่าขน\nเปิดหรือปิด ตามชอบ', Icons.accessibility_new_outlined, () => setState(() => _wordEnabled = !_wordEnabled))),
                  ],
                ),
                const SizedBox(height: 14),
                Row(
                  children: [
                    Expanded(child: _statusLabel()),
                    Text('$_videoCount วิดีโอ', style: const TextStyle(fontSize: 12, color: Colors.black54)),
                  ],
                ),
                _buildCamera(camera),
              ],
            );

        Widget _outlineButton(String label, IconData icon, VoidCallback onPressed) => SizedBox(
              height: 58,
              child: OutlinedButton(
                onPressed: onPressed,
                style: OutlinedButton.styleFrom(
                  foregroundColor: Colors.black,
                  side: const BorderSide(color: Colors.black, width: 2),
                  shape: const RoundedRectangleBorder(borderRadius: BorderRadius.zero),
                  padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 5),
                ),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Icon(icon, size: 18),
                    const SizedBox(width: 6),
                    Flexible(child: Text(label, textAlign: TextAlign.center, style: const TextStyle(fontSize: 12, height: 1.15))),
                  ],
                ),
              ),
            );

        Widget _statusLabel() => Row(
              children: [
                Icon(Icons.circle, size: 10, color: _status.contains('ไม่ได้') ? Colors.red : Colors.green),
                const SizedBox(width: 6),
                Flexible(child: Text(_status, style: const TextStyle(fontSize: 12), overflow: TextOverflow.ellipsis)),
              ],
    }
    final camera = widget.cameras.firstWhere(
      (item) => item.lensDirection == CameraLensDirection.front,
      orElse: () => widget.cameras.first,
    );
    final controller = CameraController(camera, ResolutionPreset.medium, enableAudio: false);
    await controller.initialize();
    if (!mounted) return;
    setState(() {
      _camera = controller;
      _status = 'พร้อมแปลภาษา';
    });
    _timer = Timer.periodic(const Duration(milliseconds: 850), (_) => _analyze());
  }

  Future<void> _analyze() async {
    final camera = _camera;
    if (_busy || _recording || camera == null || !camera.value.isInitialized) return;
    _busy = true;
    try {
      final image = await camera.takePicture();
      final bytes = await image.readAsBytes();
      final response = await http.post(
        Uri.parse('$backendUrl/predict'),
        headers: {'Content-Type': 'application/octet-stream'},
        body: bytes,
      ).timeout(const Duration(seconds: 4));
      if (response.statusCode != 200) throw Exception('backend ${response.statusCode}');
      final result = jsonDecode(response.body) as Map<String, dynamic>;
      if (!mounted) return;
      setState(() {
        _lastFrame = bytes;
        _letter = _characterEnabled ? (result['character'] as String?) ?? '-' : '-';
        _word = _wordEnabled ? (result['word'] as String?) ?? '-' : '-';
        _status = result['hand_present'] == true ? 'ตรวจพบมือ' : 'รอท่ามือ';
        _handCount = (result['hand_count'] as num?)?.toInt() ?? 0;
        _facePresent = result['face_present'] == true;
      });
    } catch (_) {
      if (mounted) setState(() => _status = 'เชื่อมต่อ AI ไม่ได้');
    } finally {
      _busy = false;
    }
  }

  void _addLetter() {
    if (_letter != '-') setState(() => _written += _letter);
  }

  void _clearWritten() => setState(() => _written = '');

  void _deleteLast() {
    if (_written.isNotEmpty) setState(() => _written = _written.substring(0, _written.length - 1));
  }

  Future<void> _speakResult() async {
    final text = _word != '-' ? _word : _written;
    if (text.isNotEmpty) {
      await _tts.setLanguage('th-TH');
      await _tts.speak(text);
    }
  }

  Future<void> _editWritten() async {
    final controller = TextEditingController(text: _written);
    final value = await showDialog<String>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('แก้ไขผลลัพธ์'),
        content: TextField(controller: controller, autofocus: true),
        actions: [
          TextButton(onPressed: () => Navigator.pop(context), child: const Text('ยกเลิก')),
          FilledButton(onPressed: () => Navigator.pop(context, controller.text), child: const Text('บันทึก')),
        ],
      ),
    );
    controller.dispose();
    if (value != null) setState(() => _written = value.trim());
  }

  Future<void> _trainWord() async {
    final controller = TextEditingController();
    final word = await showDialog<String>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('ฝึกคำจากภาพปัจจุบัน'),
        content: TextField(controller: controller, autofocus: true, decoration: const InputDecoration(hintText: 'เช่น สวัสดี')),
        actions: [
          TextButton(onPressed: () => Navigator.pop(context), child: const Text('ยกเลิก')),
          FilledButton(onPressed: () => Navigator.pop(context, controller.text), child: const Text('ฝึกคำ')),
        ],
      ),
    );
    controller.dispose();
    if (word == null || word.trim().isEmpty || _lastFrame == null) return;
    try {
      final response = await http.post(Uri.parse('$backendUrl/train'), headers: {'x-word': word.trim()}, body: _lastFrame);
      if (!mounted) return;
      setState(() => _status = response.statusCode == 200 ? 'ฝึกคำสำเร็จ' : 'ฝึกคำไม่สำเร็จ');
    } catch (_) {
      if (mounted) setState(() => _status = 'เชื่อมต่อ AI ไม่ได้');
    }
  }

  Future<void> _syncVideos() async {
    try {
      final response = await http.get(Uri.parse('$backendUrl/videos')).timeout(const Duration(seconds: 5));
      if (response.statusCode == 200 && mounted) {
        setState(() => _videoCount = (jsonDecode(response.body) as List<dynamic>).length);
      }
    } catch (_) {}
  }

  Future<void> _recordTrainingVideo() async {
    final camera = _camera;
    if (camera == null || !camera.value.isInitialized || _recording) return;
    final controller = TextEditingController();
    final word = await showDialog<String>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('อัดวิดีโอฝึกคำ'),
        content: TextField(controller: controller, autofocus: true, decoration: const InputDecoration(hintText: 'เช่น สวัสดี')),
        actions: [
          TextButton(onPressed: () => Navigator.pop(context), child: const Text('ยกเลิก')),
          FilledButton(onPressed: () => Navigator.pop(context, controller.text), child: const Text('เริ่มอัด')),
        ],
      ),
    );
    controller.dispose();
    if (word == null || word.trim().isEmpty) return;
    try {
      await camera.startVideoRecording();
      setState(() {
        _recording = true;
        _status = 'กำลังอัดวิดีโอ 4 วินาที';
      });
      await Future<void>.delayed(const Duration(seconds: 4));
      final video = await camera.stopVideoRecording();
      final response = await http.post(
        Uri.parse('$backendUrl/train-video'),
        headers: {'x-word': word.trim(), 'Content-Type': 'video/mp4'},
        body: await video.readAsBytes(),
      ).timeout(const Duration(minutes: 3));
      if (!mounted) return;
      setState(() {
        _recording = false;
        _status = response.statusCode == 200 ? 'ซิงค์วิดีโอและฝึกสำเร็จ' : 'ฝึกวิดีโอไม่สำเร็จ';
      });
      await _syncVideos();
    } catch (_) {
      if (mounted) setState(() {
        _recording = false;
        _status = 'อัดหรือซิงค์วิดีโอไม่สำเร็จ';
      });
    }
  }

  Future<void> _playWordVideo() async {
    final word = _word == '-' ? _written : _word;
    if (word == '-') return;
    await _video?.dispose();
    final controller = VideoPlayerController.networkUrl(
      Uri.parse('$backendUrl/video/${Uri.encodeComponent(word)}'),
    );
    try {
      await controller.initialize();
      await controller.play();
      if (mounted) setState(() => _video = controller);
    } catch (_) {
      await controller.dispose();
      if (mounted) setState(() => _status = 'ยังไม่มีวิดีโอของคำนี้');
    }
  }

  @override
  void dispose() {
    _timer?.cancel();
    _camera?.dispose();
    _video?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final camera = _camera;
    return Scaffold(
      appBar: AppBar(
        title: const Text('ภาษามือไทย', style: TextStyle(fontWeight: FontWeight.w800)),
        actions: [
          IconButton(
            tooltip: 'สลับโหมดคำศัพท์',
            onPressed: () => setState(() => _wordMode = !_wordMode),
            icon: Icon(_wordMode ? Icons.translate : Icons.abc),
          ),
          IconButton(tooltip: 'ตรวจตัวอักษร', onPressed: () => setState(() => _characterEnabled = !_characterEnabled), icon: Icon(_characterEnabled ? Icons.abc : Icons.abc_outlined)),
          IconButton(tooltip: 'ตรวจคำ', onPressed: () => setState(() => _wordEnabled = !_wordEnabled), icon: Icon(_wordEnabled ? Icons.translate : Icons.translate_outlined)),
        ],
      ),
      body: ListView(
        padding: const EdgeInsets.fromLTRB(16, 8, 16, 28),
        children: [
          _buildCamera(camera),
          const SizedBox(height: 14),
          _buildStatus(),
          const SizedBox(height: 6),
          Text('วิดีโอที่ซิงค์แล้ว $_videoCount รายการ', style: TextStyle(color: Colors.grey.shade700)),
          const SizedBox(height: 12),
          _buildResult(),
          const SizedBox(height: 12),
          _buildWrittenBox(),
          const SizedBox(height: 12),
          _buildActions(),
          if (_video != null) ...[
            const SizedBox(height: 16),
            AspectRatio(aspectRatio: _video!.value.aspectRatio, child: VideoPlayer(_video!)),
          ],
        ],
      ),
    );
  }

  Widget _buildCamera(CameraController? camera) {
    return ClipRRect(
      borderRadius: BorderRadius.circular(20),
      child: AspectRatio(
        aspectRatio: 4 / 3,
        child: camera?.value.isInitialized == true
            ? CameraPreview(camera!)
            : (_lastFrame == null
                ? Container(color: const Color(0xff12383d), child: const Center(child: CircularProgressIndicator(color: Colors.white)))
                : Image.memory(_lastFrame!, fit: BoxFit.cover)),
      ),
    );
  }

  Widget _buildStatus() => Row(
        children: [
          Icon(Icons.circle, size: 12, color: _status.contains('ไม่ได้') ? Colors.red : Colors.green),
          const SizedBox(width: 8),
          Expanded(child: Text(_status, style: const TextStyle(fontWeight: FontWeight.w700), overflow: TextOverflow.ellipsis)),
          const SizedBox(width: 8),
          Text('มือ $_handCount/2  ${_facePresent ? 'หน้า OK' : 'รอหน้า'}', style: TextStyle(color: Colors.grey.shade700)),
        ],
      );

  Widget _buildResult() => Container(
        padding: const EdgeInsets.all(18),
        decoration: BoxDecoration(color: const Color(0xff12383d), borderRadius: BorderRadius.circular(20)),
        child: Row(
          children: [
            Expanded(child: _resultItem('ท่าปัจจุบัน', _letter, Icons.pan_tool_alt)),
            Container(width: 1, height: 52, color: Colors.white24),
            Expanded(child: _resultItem('คำที่ตรวจพบ', _word, Icons.chat_bubble_outline)),
          ],
        ),
      );

  Widget _resultItem(String label, String value, IconData icon) => Column(
        children: [
          Icon(icon, color: const Color(0xfff5c86b), size: 22),
          const SizedBox(height: 5),
          Text(label, style: const TextStyle(color: Colors.white70, fontSize: 12)),
          Text(value, style: const TextStyle(color: Colors.white, fontSize: 26, fontWeight: FontWeight.w800)),
        ],
      );

  Widget _buildWrittenBox() => Row(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Expanded(
            child: Container(
              constraints: const BoxConstraints(minHeight: 58),
              padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
              decoration: BoxDecoration(border: Border.all(color: Colors.black, width: 2)),
              child: Text(
                _written.isEmpty ? 'คำที่แสดงผลออกมาตามภาษามือ\nหรือ ประโยคที่สามารถเขียนและลบได้' : _written,
                style: const TextStyle(fontSize: 14, height: 1.35, color: Colors.black),
              ),
            ),
          ),
          const SizedBox(width: 8),
          SizedBox(
            width: 58,
            child: OutlinedButton(
              onPressed: _speakResult,
              style: OutlinedButton.styleFrom(
                foregroundColor: Colors.black,
                side: const BorderSide(color: Colors.black, width: 2),
                shape: const RoundedRectangleBorder(borderRadius: BorderRadius.zero),
                padding: EdgeInsets.zero,
              ),
              child: const Text('กดอ่าน', style: TextStyle(fontSize: 13)),
            ),
          ),
        ],
      );

  Widget _buildActions() => Wrap(
        spacing: 10,
        runSpacing: 10,
        children: [
          _actionButton(_addLetter, Icons.add, 'เพิ่มผลลัพธ์', true),
          _actionButton(_speakResult, Icons.volume_up_outlined, 'พูดผลลัพธ์', false),
          _actionButton(_playWordVideo, Icons.play_circle_outline, 'ดูท่าคำนี้', false),
          _actionButton(_trainWord, Icons.model_training_outlined, 'ฝึกภาพ', false),
          _actionButton(_recordTrainingVideo, Icons.videocam_outlined, _recording ? 'กำลังอัด' : 'ฝึกวิดีโอ', false),
        ],
      );

  Widget _actionButton(VoidCallback? onPressed, IconData icon, String label, bool filled) {
    final child = filled
        ? FilledButton.icon(onPressed: onPressed, icon: Icon(icon), label: Text(label))
        : OutlinedButton.icon(onPressed: onPressed, icon: Icon(icon), label: Text(label));
    return SizedBox(width: 170, child: child);
  }
}
