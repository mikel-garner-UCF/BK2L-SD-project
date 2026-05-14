# Knightro Faculty Face Recognition — Privacy & Data Handling Policy

**Project:** Bring Knightro to Life (BK2L)<br>
**Subsystem:** Faculty Face Recognition <br>
**Version:** 3.0 <br>

---

## 1. Purpose

This document describes how the Knightro face recognition subsystem collects, stores, uses,
and deletes biometric data belonging to UCF faculty members who choose to enroll in the
"personalized greeting" feature. It exists to 
 give to enrolled faculty a clear, plain-language understanding of what the system does
with their data.

The guiding principle: **store the minimum amount of data needed, in the most privacy-preserving
form available, and give enrolled faculty complete control over removal.**

## 2. Scope

This policy covers:
- Face data collected during faculty enrollment
- Face data captured live at the Knightro deployment location during normal operation
- Any logs, backups, or derived files that contain face data

This policy does **not** cover:
- Audio/speech data
- General non-biometric usage logs (e.g., "number of interactions per day")

## 3. What data is collected

### 3.1 At enrollment

When a faculty member agrees to be enrolled, the admin operator runs the enrollment script,
which uses a webcam to capture 5–10 short frames of the faculty member's face from slightly
different angles. For each captured frame, the system computes a **face embedding** which is a
128 number numeric vector produced by a pre-trained neural network.

**Only the embeddings are retained.** The raw image frames are held in memory during
enrollment and discarded as soon as the embeddings are computed. No raw photographs of
enrolled faculty are written to disk at any point.

### 3.2 During live operation

When Knightro is running, its camera sees a continuous video stream of whoever is in front
of it. For each detected face, the system computes an embedding in memory, compares it
against the stored enrolled embeddings, and then **immediately discards the live embedding**.
Live camera frames and live embeddings are never written to disk.

### 3.3 What is explicitly NOT collected

- No raw images or video of enrolled faculty after enrollment completes
- No raw images or video of bystanders, visitors, or anyone who has not enrolled
- No UCF IDs or other personal identifiers of non-enrolled people
- No attempt to classify, infer, or store demographic information (age, gender, ethnicity, etc.)

## 4. Why embeddings instead of photos?

A face embedding is a list of 128 floating-point numbers that a neural network produced by
"looking at" a face. Two photos of the same person produce embeddings that are close together
in 128-dimensional space; two photos of different people produce embeddings that are far
apart. The system compares distances to decide "match" or "no match."

Crucially, the transformation from photo to embedding is **one-way**: you cannot reconstruct
a recognizable photograph from a 128-number vector. This is a meaningful privacy
improvement over storing raw images, because even if the storage file is stolen or leaked,
the attacker does not obtain photos of faculty, only numbers that are useless without
the exact same neural network and a matching face to compare against.

## 5. How data is stored

- **Location:** On the Raspberry Pi local filesystem only. No cloud storage, no network
  transmission, no backups off the device.
- **Format:** Embeddings are serialized to a structured file (JSON or Python pickle) and
  then the entire file is encrypted using the `cryptography` library's Fernet symmetric
  encryption before being written to disk.
- **Key storage:** The encryption key is stored separately from the encrypted data file in a restricted-permission key file or environment variable. An attacker who only
  obtains the encrypted embeddings file cannot decrypt it without also obtaining the key.
- **Filesystem permissions:** The encrypted file is set to mode 600 (read/write for the
  owner only) so that only the Pi's designated user account can read it.
- **Encryption at rest:** Yes (Fernet / AES-128-CBC + HMAC-SHA256 under the hood).
- **Encryption in transit:** Not applicable since data never leaves the device.

## 6. Who has access

- **Enrolled faculty:** May request to see the list of enrolled names at any time, and may
  request immediate removal of their own data at any time.
- **Admin operator (a designated project team member):** Can run the enrollment and removal
  scripts. Has access to the encryption key.
- **Other team members:** Do not have access to the encryption key and therefore cannot read
  stored embeddings.
- **The Knightro runtime process:** Loads the decrypted embeddings into memory only for the
  purpose of comparing against live faces. Does not log, export, or transmit them.

## 7. Enrollment procedure

1. The faculty member reads and signs the consent form sent by email previously to the meeting.
2. The admin operator runs `python src/enroll.py --name "Dr. Smith"`.
3. The script opens the webcam and guides capture of 5–10 frames from different angles.
4. The script computes embeddings in memory and discards the raw frames.
5. The script adds the new embeddings to the encrypted on-device database.
6. The admin operator confirms enrollment success with the faculty member.

## 8. Removal procedure 

A faculty member may request removal at any time, for any reason, without explanation.

1. The faculty member notifies any project team member of the removal request (verbally,
   by email, or in writing).
2. The admin operator runs `python src/enroll.py --remove --name "Dr. Smith"`.
3. The script deletes the named entry from the encrypted database and re-encrypts the file.
4. The admin operator confirms removal with the faculty member within 7 days.
5. The signed consent form for the removed individual is destroyed or marked "withdrawn."

## 9. Data retention

Enrolled embeddings are retained until **any** of the following occurs, whichever comes first:

- The faculty member requests removal (removal executed within 7 days).
- The faculty member is no longer affiliated with UCF.
- The Knightro project formally concludes and is decommissioned.

On any of these triggers, the entry is deleted from the encrypted database.

## 10. Unknown-person handling

Knightro will **not** attempt to identify, name, describe, or infer any information about
people who are not enrolled. When the recognition pipeline returns "no match" for a detected
face, Knightro treats that person as a generic visitor and gives a generic greeting only.

If a user verbally asks Knightro to identify, describe, or reveal information about any
person (enrolled or not), Knightro refuses.

## 11. Incident response

If the encrypted embeddings file or the encryption key is suspected to be compromised
(lost laptop, leaked repo, etc.), the admin operator will:

1. Immediately rotate the encryption key and re-encrypt the database.
2. Notify all currently enrolled faculty of the incident within 7 days.
3. Offer each enrolled faculty member the option to withdraw enrollment.
4. Document the incident in the project records.
