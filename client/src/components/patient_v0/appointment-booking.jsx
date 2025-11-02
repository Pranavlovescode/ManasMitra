"use client";

import { useState, useEffect } from "react";
import { Button } from "../ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "../ui/card";
import { Input } from "../ui/input";
import { Badge } from "../ui/badge";
import { Textarea } from "../ui/textarea";
import { Calendar, Clock, User, CheckCircle, XCircle, AlertCircle } from "lucide-react";

export default function AppointmentBooking({ userId }) {
  const [appointments, setAppointments] = useState([]);
  const [therapists, setTherapists] = useState([]);
  const [showBooking, setShowBooking] = useState(false);
  const [selectedTherapist, setSelectedTherapist] = useState(null);
  const [selectedDate, setSelectedDate] = useState("");
  const [selectedTime, setSelectedTime] = useState("");
  const [sessionType, setSessionType] = useState("individual");
  const [duration, setDuration] = useState(60);
  const [meetingType, setMeetingType] = useState("virtual");
  const [patientNotes, setPatientNotes] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [loadingTherapists, setLoadingTherapists] = useState(false);

  useEffect(() => {
    if (userId) {
      fetchAppointments();
    }
  }, [userId]);

  const fetchAppointments = async () => {
    try {
      const response = await fetch(`/api/appointments?role=patient`);
      if (response.ok) {
        const data = await response.json();
        setAppointments(data);
      }
    } catch (error) {
      console.error("Failed to fetch appointments:", error);
    }
  };

  const fetchTherapists = async () => {
    setLoadingTherapists(true);
    try {
      const response = await fetch('/api/therapists/available');
      if (response.ok) {
        const data = await response.json();
        console.log('📊 Therapists data:', data);
        // API returns { therapists: [...] } format
        if (data.therapists && Array.isArray(data.therapists)) {
          setTherapists(data.therapists);
        } else if (Array.isArray(data)) {
          setTherapists(data);
        } else {
          console.error('Invalid therapists data format:', data);
          setTherapists([]);
        }
      }
    } catch (error) {
      console.error("Failed to fetch therapists:", error);
      setTherapists([]);
    } finally {
      setLoadingTherapists(false);
    }
  };

  const handleShowBooking = () => {
    setShowBooking(true);
    fetchTherapists();
  };

  const handleBookAppointment = async () => {
    if (!selectedTherapist) {
      alert("Please select a therapist");
      return;
    }
    if (!selectedDate || !selectedTime) {
      alert("Please select date and time");
      return;
    }

    setIsLoading(true);

    try {
      const scheduledDateTime = new Date(`${selectedDate}T${selectedTime}`);
      
      const response = await fetch("/api/appointments", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          therapistId: selectedTherapist._id,
          scheduledAt: scheduledDateTime.toISOString(),
          type: sessionType,
          duration: duration,
          meetingType: meetingType,
          patientNotes: patientNotes,
        }),
      });

      if (response.ok) {
        alert("Appointment request sent successfully! Waiting for therapist approval.");
        setShowBooking(false);
        setSelectedTherapist(null);
        setSelectedDate("");
        setSelectedTime("");
        setPatientNotes("");
        fetchAppointments();
      } else {
        const error = await response.json();
        alert(error.error || "Failed to book appointment");
      }
    } catch (error) {
      console.error("Failed to book appointment:", error);
      alert("Failed to book appointment. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'pending': return 'bg-yellow-100 text-yellow-800';
      case 'accepted': return 'bg-green-100 text-green-800';
      case 'declined': return 'bg-red-100 text-red-800';
      case 'completed': return 'bg-blue-100 text-blue-800';
      case 'cancelled': return 'bg-gray-100 text-gray-800';
      case 'rescheduled': return 'bg-purple-100 text-purple-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'accepted': return <CheckCircle className="w-4 h-4" />;
      case 'declined': return <XCircle className="w-4 h-4" />;
      case 'pending': return <AlertCircle className="w-4 h-4" />;
      default: return null;
    }
  };

  return (
    <div className="space-y-6">
      {!showBooking ? (
        <Card>
          <CardHeader>
            <CardTitle>Book an Appointment</CardTitle>
            <CardDescription>
              Schedule a therapy session with a therapist
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Button onClick={handleShowBooking} className="w-full">
              Book New Appointment
            </Button>
          </CardContent>
        </Card>
      ) : (
        <Card>
          <CardHeader>
            <CardTitle>Schedule Your Session</CardTitle>
            <CardDescription>Select a therapist and choose your preferred time</CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Therapist Selection */}
            <div className="space-y-3">
              <label className="text-sm font-medium">Select Therapist</label>
              {loadingTherapists ? (
                <div className="text-center py-4 text-gray-500">Loading therapists...</div>
              ) : therapists.length === 0 ? (
                <div className="text-center py-4 text-gray-500">No therapists available</div>
              ) : (
                <div className="grid gap-3 max-h-64 overflow-y-auto">
                  {therapists.map((therapist) => (
                    <div
                      key={therapist._id}
                      onClick={() => setSelectedTherapist(therapist)}
                      className={`p-4 border-2 rounded-lg cursor-pointer transition-all ${
                        selectedTherapist?._id === therapist._id
                          ? 'border-indigo-600 bg-indigo-50'
                          : 'border-gray-200 hover:border-indigo-300'
                      }`}
                    >
                      <div className="flex items-start justify-between">
                        <div className="flex-1">
                          <div className="flex items-center gap-2">
                            <User className="w-4 h-4 text-gray-500" />
                            <p className="font-semibold">{therapist.name}</p>
                          </div>
                          <p className="text-sm text-gray-600 mt-1">{therapist.email}</p>
                          {therapist.yearsOfExperience > 0 && (
                            <p className="text-sm text-gray-500 mt-1">
                              {therapist.yearsOfExperience} years experience
                            </p>
                          )}
                          {therapist.specializations?.length > 0 && (
                            <div className="flex flex-wrap gap-1 mt-2">
                              {therapist.specializations.slice(0, 3).map((spec, idx) => (
                                <Badge key={idx} variant="secondary" className="text-xs">
                                  {spec}
                                </Badge>
                              ))}
                            </div>
                          )}
                        </div>
                        {selectedTherapist?._id === therapist._id && (
                          <CheckCircle className="w-5 h-5 text-indigo-600" />
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {selectedTherapist && (
              <>
                {/* Session Type */}
                <div className="space-y-2">
                  <label className="text-sm font-medium">Session Type</label>
                  <div className="flex gap-4">
                    <label className="flex items-center gap-2 cursor-pointer">
                      <input
                        type="radio"
                        value="individual"
                        checked={sessionType === "individual"}
                        onChange={(e) => setSessionType(e.target.value)}
                      />
                      <span className="text-sm">Individual</span>
                    </label>
                    <label className="flex items-center gap-2 cursor-pointer">
                      <input
                        type="radio"
                        value="couple"
                        checked={sessionType === "couple"}
                        onChange={(e) => setSessionType(e.target.value)}
                      />
                      <span className="text-sm">Couple</span>
                    </label>
                    <label className="flex items-center gap-2 cursor-pointer">
                      <input
                        type="radio"
                        value="family"
                        checked={sessionType === "family"}
                        onChange={(e) => setSessionType(e.target.value)}
                      />
                      <span className="text-sm">Family</span>
                    </label>
                  </div>
                </div>

                {/* Meeting Type */}
                <div className="space-y-2">
                  <label className="text-sm font-medium">Meeting Type</label>
                  <div className="flex gap-4">
                    <label className="flex items-center gap-2 cursor-pointer">
                      <input
                        type="radio"
                        value="virtual"
                        checked={meetingType === "virtual"}
                        onChange={(e) => setMeetingType(e.target.value)}
                      />
                      <span className="text-sm">Virtual</span>
                    </label>
                    <label className="flex items-center gap-2 cursor-pointer">
                      <input
                        type="radio"
                        value="in-person"
                        checked={meetingType === "in-person"}
                        onChange={(e) => setMeetingType(e.target.value)}
                      />
                      <span className="text-sm">In-Person</span>
                    </label>
                  </div>
                </div>

                {/* Date and Time */}
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <label className="text-sm font-medium flex items-center gap-2">
                      <Calendar className="w-4 h-4" />
                      Date
                    </label>
                    <Input
                      type="date"
                      value={selectedDate}
                      onChange={(e) => setSelectedDate(e.target.value)}
                      min={new Date().toISOString().split('T')[0]}
                    />
                  </div>

                  <div className="space-y-2">
                    <label className="text-sm font-medium flex items-center gap-2">
                      <Clock className="w-4 h-4" />
                      Time
                    </label>
                    <Input
                      type="time"
                      value={selectedTime}
                      onChange={(e) => setSelectedTime(e.target.value)}
                    />
                  </div>
                </div>

                {/* Duration */}
                <div className="space-y-2">
                  <label className="text-sm font-medium">Preferred Duration</label>
                  <select
                    value={duration}
                    onChange={(e) => setDuration(Number(e.target.value))}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md"
                  >
                    <option value={30}>30 minutes</option>
                    <option value={45}>45 minutes</option>
                    <option value={60}>60 minutes</option>
                    <option value={90}>90 minutes</option>
                  </select>
                </div>

                {/* Notes */}
                <div className="space-y-2">
                  <label className="text-sm font-medium">Notes (Optional)</label>
                  <Textarea
                    value={patientNotes}
                    onChange={(e) => setPatientNotes(e.target.value)}
                    placeholder="Any specific concerns or topics you'd like to discuss..."
                    rows={3}
                  />
                </div>

                <div className="flex gap-2">
                  <Button
                    onClick={() => {
                      setShowBooking(false);
                      setSelectedTherapist(null);
                    }}
                    variant="outline"
                    className="flex-1"
                  >
                    Cancel
                  </Button>
                  <Button
                    onClick={handleBookAppointment}
                    disabled={isLoading}
                    className="flex-1"
                  >
                    {isLoading ? "Booking..." : "Book Appointment"}
                  </Button>
                </div>
              </>
            )}
          </CardContent>
        </Card>
      )}

      {appointments.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Your Appointments</CardTitle>
            <CardDescription>View and manage your scheduled sessions</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {appointments.map((apt) => (
                <div key={apt._id} className="p-4 bg-gray-50 rounded-lg border border-gray-200">
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-2">
                        <User className="w-4 h-4 text-gray-500" />
                        <p className="font-semibold">
                          Dr. {apt.therapistId?.userId?.firstName} {apt.therapistId?.userId?.lastName}
                        </p>
                      </div>
                      <div className="space-y-1 text-sm text-gray-600">
                        <p className="flex items-center gap-2">
                          <Calendar className="w-3 h-3" />
                          {new Date(apt.scheduledAt).toLocaleDateString('en-US', {
                            weekday: 'long',
                            year: 'numeric',
                            month: 'long',
                            day: 'numeric'
                          })}
                        </p>
                        <p className="flex items-center gap-2">
                          <Clock className="w-3 h-3" />
                          {new Date(apt.scheduledAt).toLocaleTimeString('en-US', {
                            hour: '2-digit',
                            minute: '2-digit'
                          })} • {apt.duration} minutes
                        </p>
                        <p className="capitalize">
                          {apt.type} Session • {apt.meetingType}
                        </p>
                      </div>
                      {apt.patientNotes && (
                        <p className="text-sm text-gray-500 mt-2 italic">
                          Your notes: {apt.patientNotes}
                        </p>
                      )}
                      {apt.therapistNotes && apt.status === 'declined' && (
                        <p className="text-sm text-red-600 mt-2">
                          Therapist note: {apt.therapistNotes}
                        </p>
                      )}
                    </div>
                    <div className="flex flex-col items-end gap-2">
                      <Badge className={`${getStatusColor(apt.status)} flex items-center gap-1`}>
                        {getStatusIcon(apt.status)}
                        {apt.status}
                      </Badge>
                      {apt.status === 'pending' && (
                        <span className="text-xs text-gray-500">Awaiting approval</span>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
