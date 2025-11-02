import mongoose from "mongoose";

const MONGODB_URI = process.env.MONGODB_URI;

if (!MONGODB_URI) {
  throw new Error(
    "Please define the MONGODB_URI environment variable inside .env.local"
  );
}

/**
 * Global is used here to maintain a cached connection across hot reloads
 * in development. This prevents connections growing exponentially
 * during API Route usage.
 */
let cached = global.mongoose;

if (!cached) {
  cached = global.mongoose = { conn: null, promise: null };
}

async function connectDB() {
  if (cached.conn && cached.conn.readyState === 1) {
    return cached.conn;
  }

  if (!cached.promise) {
    const opts = {
      bufferCommands: false,
      maxPoolSize: 10,
      serverSelectionTimeoutMS: 10000,
      socketTimeoutMS: 45000,
      connectTimeoutMS: 10000,
      family: 4, // Use IPv4, skip trying IPv6
    };

    try {
      console.log("Connecting to MongoDB...");
      cached.promise = mongoose.connect(MONGODB_URI, opts).then((mongoose) => {
        console.log("MongoDB connected successfully");
        return mongoose;
      });
    } catch (error) {
      console.error("MongoDB connection error:", error);
      cached.promise = null;
      throw new Error(`MongoDB connection failed: ${error.message}`);
    }
  }

  try {
    cached.conn = await cached.promise;
  } catch (e) {
    console.error("Error establishing MongoDB connection:", e);
    cached.promise = null;

    // Check if it's a network connectivity issue
    if (
      e.code === "ECONNREFUSED" ||
      e.code === "EAI_AGAIN" ||
      e.syscall === "querySrv"
    ) {
      throw new Error(
        "Unable to connect to MongoDB. Please check your internet connection and database configuration."
      );
    }

    throw new Error(`Failed to establish MongoDB connection: ${e.message}`);
  }

  return cached.conn;
}

export default connectDB;
