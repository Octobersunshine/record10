const express = require("express");
const http = require("http");
const { Server } = require("socket.io");

const app = express();
const path = require("path");

app.use(express.static(path.join(__dirname, "public")));

const server = http.createServer(app);
const io = new Server(server, {
  cors: { origin: "*", methods: ["GET", "POST"] },
});

const HEARTBEAT_INTERVAL = 5000;
const HEARTBEAT_TIMEOUT = 15000;

const rooms = new Map();

function getRoom(roomId) {
  if (!rooms.has(roomId)) {
    rooms.set(roomId, { users: new Map() });
  }
  return rooms.get(roomId);
}

function leaveRoom(socket) {
  const roomId = socket.data.roomId;
  if (!roomId) return;

  const room = rooms.get(roomId);
  if (!room) return;

  const userId = socket.data.userId;
  room.users.delete(userId);

  socket.to(roomId).emit("user-left", { userId, roomId });

  if (room.users.size === 0) {
    rooms.delete(roomId);
  }

  socket.leave(roomId);
  socket.data.roomId = null;
  socket.data.userId = null;
}

function validateRoomMessage(socket, roomId, targetUserId) {
  if (!roomId) {
    console.log(`[VALIDATE] ${socket.data.userId || socket.id}: missing roomId`);
    return false;
  }

  if (socket.data.roomId !== roomId) {
    console.log(
      `[VALIDATE] ${socket.data.userId || socket.id}: room mismatch, socket.roomId=${socket.data.roomId}, message.roomId=${roomId}`
    );
    return false;
  }

  const room = rooms.get(roomId);
  if (!room) {
    console.log(`[VALIDATE] ${socket.data.userId || socket.id}: room ${roomId} not found`);
    return false;
  }

  if (!room.users.has(socket.data.userId)) {
    console.log(
      `[VALIDATE] ${socket.data.userId || socket.id}: sender not in room ${roomId}`
    );
    return false;
  }

  if (targetUserId && !room.users.has(targetUserId)) {
    console.log(
      `[VALIDATE] ${socket.data.userId || socket.id}: target ${targetUserId} not in room ${roomId}`
    );
    return false;
  }

  return true;
}

function checkHeartbeatTimeouts() {
  const now = Date.now();
  rooms.forEach((room, roomId) => {
    room.users.forEach((userData, userId) => {
      if (now - userData.lastHeartbeat > HEARTBEAT_TIMEOUT) {
        const socketId = userData.socketId;
        const socket = io.sockets.sockets.get(socketId);
        console.log(
          `[TIMEOUT] User ${userId} in room ${roomId} timed out, last heartbeat: ${now - userData.lastHeartbeat}ms ago`
        );
        if (socket) {
          socket.emit("kicked", { reason: "heartbeat_timeout", roomId });
          leaveRoom(socket);
        } else {
          room.users.delete(userId);
          if (room.users.size === 0) {
            rooms.delete(roomId);
          }
          io.to(roomId).emit("user-left", { userId, roomId });
        }
      }
    });
  });
}

setInterval(checkHeartbeatTimeouts, HEARTBEAT_INTERVAL);

io.on("connection", (socket) => {
  console.log(`[CONNECT] ${socket.id}`);

  socket.on("join-room", ({ roomId, userId }, callback) => {
    if (!roomId || !userId) {
      return callback?.({ success: false, error: "roomId and userId are required" });
    }

    const room = getRoom(roomId);

    if (room.users.size >= 2) {
      return callback?.({ success: false, error: "Room is full (max 2 users)" });
    }

    if (room.users.has(userId)) {
      return callback?.({ success: false, error: "userId already taken in this room" });
    }

    socket.data.roomId = roomId;
    socket.data.userId = userId;

    room.users.set(userId, {
      socketId: socket.id,
      lastHeartbeat: Date.now(),
    });
    socket.join(roomId);

    const existingUsers = [];
    room.users.forEach((userData, uid) => {
      if (uid !== userId) {
        existingUsers.push({ userId: uid, socketId: userData.socketId });
      }
    });

    socket.to(roomId).emit("user-joined", { userId, socketId: socket.id, roomId });

    callback?.({
      success: true,
      users: existingUsers,
      heartbeatInterval: HEARTBEAT_INTERVAL,
    });

    console.log(`[JOIN] ${userId} joined room ${roomId} (${room.users.size} users)`);
  });

  socket.on("ping", ({ roomId }, callback) => {
    if (!validateRoomMessage(socket, roomId)) {
      return callback?.({ success: false, error: "invalid room" });
    }
    const room = rooms.get(roomId);
    if (room && room.users.has(socket.data.userId)) {
      room.users.get(socket.data.userId).lastHeartbeat = Date.now();
    }
    callback?.({ success: true, timestamp: Date.now() });
  });

  socket.on("offer", ({ roomId, targetUserId, sdp }) => {
    if (!validateRoomMessage(socket, roomId, targetUserId)) return;

    const room = rooms.get(roomId);
    const targetSocketId = room.users.get(targetUserId).socketId;

    io.to(targetSocketId).emit("offer", {
      fromUserId: socket.data.userId,
      roomId,
      sdp,
    });

    console.log(`[OFFER] ${socket.data.userId} -> ${targetUserId} in room ${roomId}`);
  });

  socket.on("answer", ({ roomId, targetUserId, sdp }) => {
    if (!validateRoomMessage(socket, roomId, targetUserId)) return;

    const room = rooms.get(roomId);
    const targetSocketId = room.users.get(targetUserId).socketId;

    io.to(targetSocketId).emit("answer", {
      fromUserId: socket.data.userId,
      roomId,
      sdp,
    });

    console.log(`[ANSWER] ${socket.data.userId} -> ${targetUserId} in room ${roomId}`);
  });

  socket.on("ice-candidate", ({ roomId, targetUserId, candidate }) => {
    if (!validateRoomMessage(socket, roomId, targetUserId)) return;

    const room = rooms.get(roomId);
    const targetSocketId = room.users.get(targetUserId).socketId;

    io.to(targetSocketId).emit("ice-candidate", {
      fromUserId: socket.data.userId,
      roomId,
      candidate,
    });
  });

  socket.on("leave-room", ({ roomId }) => {
    if (roomId && socket.data.roomId !== roomId) {
      console.log(
        `[LEAVE] roomId mismatch: socket.roomId=${socket.data.roomId}, requested=${roomId}`
      );
      return;
    }
    const userId = socket.data.userId;
    const actualRoomId = socket.data.roomId;
    leaveRoom(socket);
    if (userId && actualRoomId) {
      console.log(`[LEAVE] ${userId} left room ${actualRoomId}`);
    }
  });

  socket.on("disconnect", (reason) => {
    const userId = socket.data.userId;
    const roomId = socket.data.roomId;
    leaveRoom(socket);
    if (userId && roomId) {
      console.log(`[DISCONNECT] ${userId} from room ${roomId} (reason: ${reason})`);
    }
  });
});

const PORT = process.env.PORT || 3000;
server.listen(PORT, () => {
  console.log(`WebRTC Signaling Server running on http://localhost:${PORT}`);
});
