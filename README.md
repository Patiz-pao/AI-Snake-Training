# AI-Snake-Training 🐍

โปรเจคนี้เป็นการพัฒนาปัญญาประดิษฐ์สำหรับการเล่นเกมงู โดยใช้ Neural Network ในการเรียนรู้และ Pygame ในการสร้างตัวเกม

## ✨ คุณสมบัติหลัก

- ระบบเกมงูพื้นฐานที่พัฒนาด้วย Pygame
- การเรียนรู้ด้วย Neural Network แบบ Feed-forward
- ระบบการเทรนโมเดลแบบอัตโนมัติ
- การแสดงผลสถิติและประสิทธิภาพของ AI ในแต่ละรอบการเรียนรู้
- การบันทึกและโหลดโมเดลที่เทรนแล้ว

## 🛠 การติดตั้ง

```bash
# Clone repository
git clone https://github.com/yourusername/AI-Snake-Training.git

# เข้าไปยังโฟลเดอร์โปรเจค
cd AI-Snake-Training

# ติดตั้ง dependencies
pip install -r requirements.txt
```

## 🎮 วิธีการใช้งาน

```python
python train.py
```

## 🧠 โครงสร้าง Neural Network

โมเดลใช้ Feed-forward Neural Network ที่มีโครงสร้างดังนี้:
- Input Layer: 11 neurons (สำหรับรับข้อมูลสภาพแวดล้อมรอบๆ งู)
- Hidden Layer: 256 neurons
- Output Layer: 3 neurons (สำหรับทิศทางการเคลื่อนที่)
