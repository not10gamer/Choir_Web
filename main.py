
import librosa
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status
from pydantic import BaseModel
import soundfile as sf
import io
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta, timezone

# --- Configuration ---
SECRET_KEY = "your-super-secret-key"  # Replace with a real secret key
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# --- FastAPI App Initialization ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Security and Authentication ---
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# --- Pydantic Models ---
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: str | None = None
    scopes: list[str] = []

class User(BaseModel):
    username: str
    scopes: list[str] = []

class UserInDB(User):
    hashed_password: str

class AnalysisResult(BaseModel):
    filename: str
    tempo: float
    pitch: str
    key: str
    chromagram: list
    spectral_centroid: float
    zero_crossing_rate: float
    spectral_contrast: list
    beats: list

# --- "Database" and Helper Functions ---
fake_users_db = {
    "user": {
        "username": "user",
        "hashed_password": pwd_context.hash("password"),
        "scopes": ["user"],
    },
    "admin": {
        "username": "admin",
        "hashed_password": pwd_context.hash("admin"),
        "scopes": ["admin", "user"],
    },
}

def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)

def authenticate_user(username, password):
    user = get_user(fake_users_db, username)
    if not user or not pwd_context.verify(password, user.hashed_password):
        return False
    return user

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        scopes = payload.get("scopes", [])
        token_data = TokenData(username=username, scopes=scopes)
    except JWTError:
        raise credentials_exception
    user = get_user(fake_users_db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

# --- API Endpoints ---
@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username, "scopes": user.scopes},
        expires_delta=access_token_expires,
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user

@app.post("/analyze/", response_model=AnalysisResult)
async def analyze_music(
    file: UploadFile = File(...), current_user: User = Depends(get_current_user)
):
    if "user" not in current_user.scopes:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    try:
        contents = await file.read()
        with io.BytesIO(contents) as bio:
            y, sr = sf.read(bio)
            if y.ndim > 1:
                y = librosa.to_mono(y.T)

        if np.all(y == 0):
            return {
                "filename": file.filename, "tempo": 0, "pitch": "N/A",
                "key": "N/A", "chromagram": [0]*12,
                "spectral_centroid": 0, "zero_crossing_rate": 0,
                "spectral_contrast": [0]*7, "beats": [],
            }

        # Existing analysis
        tempo_array, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units='frames')
        tempo = tempo_array.item() if isinstance(tempo_array, np.ndarray) else tempo_array
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)

        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        if pitches.size > 0:
            dominant_pitch_freq = np.median(pitches[magnitudes > np.median(magnitudes)])
            dominant_pitch_note = librosa.hz_to_note(dominant_pitch_freq) if dominant_pitch_freq > 0 else "N/A"
        else:
            dominant_pitch_note = "N/A"

        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        key_idx = np.argmax(chroma_mean)
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        key = notes[key_idx]
        chromagram_visual = chroma_mean.tolist()

        # New analysis
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
        spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr), axis=1)

        return {
            "filename": file.filename,
            "tempo": round(tempo, 2),
            "pitch": dominant_pitch_note,
            "key": f"{key} Major/Minor",
            "chromagram": chromagram_visual,
            "spectral_centroid": round(spectral_centroid, 2),
            "zero_crossing_rate": round(zero_crossing_rate, 4),
            "spectral_contrast": spectral_contrast.tolist(),
            "beats": beat_times.tolist(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

