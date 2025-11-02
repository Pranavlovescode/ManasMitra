import { auth, currentUser } from "@clerk/nextjs/server";
import { NextResponse } from "next/server";
import connectDB from "@/lib/mongodb";
import User from "@/models/User";
import Patient from "@/models/Patient";

// GET /api/patients - Get current patient details
export async function GET(request) {
  try {
    console.log("GET /api/patients - Starting authentication check");

    // Get auth using auth method
    const { userId } = auth();
    console.log("Auth userId:", userId);

    if (!userId) {
      console.log("No userId found - returning unauthorized");
      return NextResponse.json(
        {
          error: "Unauthorized",
          debug: {
            authResult,
            message: "No userId found in getAuth() result",
          },
        },
        { status: 401 }
      );
    }

    await connectDB();

    // Find the user first
    const user = await User.findOne({ clerkId: userId });

    if (!user) {
      return NextResponse.json({ error: "User not found" }, { status: 404 });
    }

    // Check if user is a patient
    if (user.role !== "patient") {
      return NextResponse.json(
        { error: "Access denied. Only patients can access this endpoint." },
        { status: 403 }
      );
    }

    // Find patient details
    let patient = await Patient.findOne({ clerkId: userId }).populate("userId");

    if (!patient) {
      // Return empty patient structure if not found
      return NextResponse.json({
        message: "Patient details not found",
        hasDetails: false,
        user: {
          firstName: user.firstName,
          lastName: user.lastName,
          email: user.email,
        },
      });
    }

    return NextResponse.json({
      hasDetails: true,
      patient,
      user: {
        firstName: user.firstName,
        lastName: user.lastName,
        email: user.email,
      },
    });
  } catch (error) {
    console.error("Error fetching patient details:", error);
    return NextResponse.json(
      { error: "Internal Server Error" },
      { status: 500 }
    );
  }
}

// POST /api/patients - Create patient details
export async function POST(req) {
  try {
    console.log("POST /api/patients - Starting authentication check");
    const authResult = getAuth(req);
    const { userId } = authResult || {};
    console.log("Auth userId:", userId);

    if (!userId) {
      console.log("No userId found - returning unauthorized");
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const body = await req.json();
    console.log("Request body received:", JSON.stringify(body, null, 2));

    await connectDB();

    // Find the user first
    const user = await User.findOne({ clerkId: userId });

    if (!user) {
      return NextResponse.json({ error: "User not found" }, { status: 404 });
    }

    // Check if user is a patient
    if (user.role !== "patient") {
      return NextResponse.json(
        { error: "Access denied. Only patients can create patient details." },
        { status: 403 }
      );
    }

    // Check if patient details already exist
    const existingPatient = await Patient.findOne({ clerkId: userId });

    if (existingPatient) {
      return NextResponse.json(
        { error: "Patient details already exist. Use PUT to update." },
        { status: 409 }
      );
    }

    // Create new patient
    const patient = new Patient({
      userId: user._id,
      clerkId: userId,
      ...body,
    });

    await patient.save();

    // Update user profile completion status
    await User.findByIdAndUpdate(user._id, { profileComplete: true });

    // Populate the userId field before returning
    await patient.populate("userId");

    return NextResponse.json(
      {
        message: "Patient details created successfully",
        patient,
      },
      { status: 201 }
    );
  } catch (error) {
    console.error("Error creating patient details:", error);
    return NextResponse.json(
      { error: "Internal Server Error" },
      { status: 500 }
    );
  }
}

// PUT /api/patients - Update patient details
export async function PUT(req) {
  try {
    const authResult = getAuth(req);
    const { userId } = authResult || {};

    if (!userId) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const body = await req.json();

    await connectDB();

    // Find the user first
    const user = await User.findOne({ clerkId: userId });

    if (!user) {
      return NextResponse.json({ error: "User not found" }, { status: 404 });
    }

    // Check if user is a patient
    if (user.role !== "patient") {
      return NextResponse.json(
        { error: "Access denied. Only patients can update patient details." },
        { status: 403 }
      );
    }

    // Find and update patient details
    let patient = await Patient.findOne({ clerkId: userId });

    if (!patient) {
      // Create new patient if doesn't exist
      patient = new Patient({
        userId: user._id,
        clerkId: userId,
        ...body,
      });
    } else {
      // Update existing patient
      Object.assign(patient, body);
    }

    await patient.save();

    // Update user profile completion status
    await User.findByIdAndUpdate(user._id, { profileComplete: true });

    // Populate the userId field before returning
    await patient.populate("userId");

    return NextResponse.json({
      message: "Patient details updated successfully",
      patient,
    });
  } catch (error) {
    console.error("Error updating patient details:", error);
    return NextResponse.json(
      { error: "Internal Server Error" },
      { status: 500 }
    );
  }
}

// DELETE /api/patients - Delete patient details (soft delete)
export async function DELETE(req) {
  try {
    const authResult = getAuth(req);
    const { userId } = authResult || {};

    if (!userId) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    await connectDB();

    // Find the user first
    const user = await User.findOne({ clerkId: userId });

    if (!user) {
      return NextResponse.json({ error: "User not found" }, { status: 404 });
    }

    // Check if user is a patient
    if (user.role !== "patient") {
      return NextResponse.json(
        { error: "Access denied. Only patients can delete patient details." },
        { status: 403 }
      );
    }

    // Soft delete by setting activePatient to false
    const patient = await Patient.findOneAndUpdate(
      { clerkId: userId },
      { "status.activePatient": false },
      { new: true }
    );

    if (!patient) {
      return NextResponse.json(
        { error: "Patient details not found" },
        { status: 404 }
      );
    }

    // Update user profile completion status
    await User.findByIdAndUpdate(user._id, { profileComplete: false });

    return NextResponse.json({
      message: "Patient details deactivated successfully",
    });
  } catch (error) {
    console.error("Error deactivating patient details:", error);
    return NextResponse.json(
      { error: "Internal Server Error" },
      { status: 500 }
    );
  }
}
