const { io } = require("socket.io-client");

const SERVER_URL = "http://localhost:3000";

function createClient(name) {
  return new Promise((resolve) => {
    const socket = io(SERVER_URL, { autoConnect: false });
    socket.name = name;

    socket.on("connect", () => {
      console.log(`[${name}] Connected: ${socket.id}`);
    });

    socket.on("disconnect", () => {
      console.log(`[${name}] Disconnected`);
    });

    socket.on("offer", (data) => {
      console.log(`[${name}] <<<< OFFER from ${data.fromUserId} (room: ${data.roomId})`);
    });

    socket.on("user-joined", (data) => {
      console.log(`[${name}] User joined: ${data.userId} (room: ${data.roomId})`);
    });

    socket.on("user-left", (data) => {
      console.log(`[${name}] User left: ${data.userId} (room: ${data.roomId})`);
    });

    socket.on("kicked", (data) => {
      console.log(`[${name}] KICKED! reason: ${data.reason} (room: ${data.roomId})`);
    });

    socket.connect();
    resolve(socket);
  });
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function testRoomIsolation() {
  console.log("\n=== Test 1: Room Isolation (消息串扰测试) ===");

  const alice = await createClient("Alice");
  const bob = await createClient("Bob");
  const charlie = await createClient("Charlie");

  await sleep(500);

  console.log("\n--- Alice joins room A ---");
  await new Promise((resolve) => {
    alice.emit("join-room", { roomId: "roomA", userId: "alice" }, (res) => {
      console.log(`[Alice] Join result:`, res);
      resolve();
    });
  });

  console.log("\n--- Bob joins room A ---");
  await new Promise((resolve) => {
    bob.emit("join-room", { roomId: "roomA", userId: "bob" }, (res) => {
      console.log(`[Bob] Join result:`, res);
      resolve();
    });
  });

  console.log("\n--- Charlie joins room B ---");
  await new Promise((resolve) => {
    charlie.emit("join-room", { roomId: "roomB", userId: "charlie" }, (res) => {
      console.log(`[Charlie] Join result:`, res);
      resolve();
    });
  });

  await sleep(500);

  console.log("\n--- Alice sends offer to Bob (correct room) ---");
  alice.emit("offer", { roomId: "roomA", targetUserId: "bob", sdp: "test-sdp-1" });

  await sleep(500);

  console.log("\n--- Alice tries to send offer to wrong room (should be rejected) ---");
  alice.emit("offer", { roomId: "roomB", targetUserId: "charlie", sdp: "test-sdp-2" });

  await sleep(500);

  console.log("\n--- Charlie tries to send offer to Bob (wrong room, should be rejected) ---");
  charlie.emit("offer", { roomId: "roomA", targetUserId: "bob", sdp: "test-sdp-3" });

  await sleep(1000);

  console.log("\n=== Test 2: Heartbeat Timeout (心跳超时测试) ===");
  const dave = await createClient("Dave");
  await sleep(500);

  console.log("\n--- Dave joins room C, but stops heartbeating ---");
  await new Promise((resolve) => {
    dave.emit("join-room", { roomId: "roomC", userId: "dave" }, (res) => {
      console.log(`[Dave] Join result:`, res);
      resolve();
    });
  });

  console.log("\nWaiting for timeout (15 seconds)...");
  for (let i = 0; i < 18; i++) {
    await sleep(1000);
    if (i % 5 === 4) console.log(`  ...${i + 1}s`);
  }

  console.log("\n=== Cleanup ===");
  alice.disconnect();
  bob.disconnect();
  charlie.disconnect();
  dave.disconnect();

  await sleep(500);
  console.log("\n=== All tests completed ===");
  process.exit(0);
}

testRoomIsolation().catch(console.error);
