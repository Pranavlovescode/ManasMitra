"use client";

import { useEffect, useState } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "../ui/card";
import { Button } from "../ui/button";
import { Badge } from "../ui/badge";
import { Input } from "../ui/input";
import { Textarea } from "../ui/textarea";
import { Calendar, Clock, User, CheckCircle, XCircle, Edit2, Trash2 } from "lucide-react";

export default function SessionManagement({ therapistId }) {
  const [appointments, setAppointments] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [filter, setFilter] = useState('all'); // 'all', 'pending', 'accepted', 'declined'
  const [editingAppointment, setEditingAppointment] = useState(null);
  const [newTime, setNewTime] = useState("");
  const [newDate, setNewDate] = useState("");
  const [newDuration, setNewDuration] = useState(60);
  const [therapistNotes, setTherapistNotes] = useState("");

  useEffect(() => {
    console.log('🔍 SessionManagement mounted with therapistId:', therapistId);
    if (therapistId) {
      fetchAppointments();
    } else {
      console.warn('⚠️ No therapistId provided to SessionManagement');
      setIsLoading(false);
      setError('No therapist ID provided');
    }
  }, [therapistId]);

  const fetchAppointments = async () => {
    setIsLoading(true);
    setError(null);
    console.log('📞 Fetching appointments for therapist...');
    
    try {
      const response = await fetch(`/api/appointments?role=therapist`);
      console.log('📊 Response status:', response.status);
      
      if (response.ok) {
        const data = await response.json();
        console.log('✅ Fetched appointments:', data.length, 'appointments');
        setAppointments(data);
      } else {
        const errorData = await response.json();
        console.error('❌ Failed to fetch:', errorData);
        setError(errorData.error || 'Failed to fetch appointments');
      }
    } catch (error) {
      console.error("❌ Error fetching appointments:", error);
      setError(error.message || 'Network error');
    } finally {
      setIsLoading(false);
    }
  };

  const handleAccept = async (appointmentId) => {
    try {
      const response = await fetch(`/api/appointments/${appointmentId}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'accept' }),
      });

      if (response.ok) {
        alert("Appointment accepted successfully!");
        fetchAppointments();
      } else {
        const error = await response.json();
        alert(error.error || "Failed to accept appointment");
      }
    } catch (error) {
      console.error("Failed to accept appointment:", error);
      alert("Failed to accept appointment");
    }
  };

  const handleDecline = async (appointmentId) => {
    const reason = prompt("Please provide a reason for declining (optional):");
    
    try {
      const response = await fetch(`/api/appointments/${appointmentId}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          action: 'decline',
          therapistNotes: reason 
        }),
      });

      if (response.ok) {
        alert("Appointment declined");
        fetchAppointments();
      } else {
        const error = await response.json();
        alert(error.error || "Failed to decline appointment");
      }
    } catch (error) {
      console.error("Failed to decline appointment:", error);
      alert("Failed to decline appointment");
    }
  };

  const handleUpdateTime = async (appointmentId) => {
    if (!newDate || !newTime) {
      alert("Please select both date and time");
      return;
    }

    try {
      const scheduledDateTime = new Date(`${newDate}T${newTime}`);
      
      const response = await fetch(`/api/appointments/${appointmentId}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          action: 'reschedule',
          scheduledAt: scheduledDateTime.toISOString()
        }),
      });

      if (response.ok) {
        alert("Appointment time updated successfully!");
        setEditingAppointment(null);
        setNewDate("");
        setNewTime("");
        fetchAppointments();
      } else {
        const error = await response.json();
        alert(error.error || "Failed to update appointment");
      }
    } catch (error) {
      console.error("Failed to update appointment:", error);
      alert("Failed to update appointment");
    }
  };

  const handleUpdateDuration = async (appointmentId) => {
    try {
      const response = await fetch(`/api/appointments/${appointmentId}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          action: 'update-duration',
          duration: newDuration
        }),
      });

      if (response.ok) {
        alert("Duration updated successfully!");
        setEditingAppointment(null);
        fetchAppointments();
      } else {
        const error = await response.json();
        alert(error.error || "Failed to update duration");
      }
    } catch (error) {
      console.error("Failed to update duration:", error);
      alert("Failed to update duration");
    }
  };

  const handleComplete = async (appointmentId) => {
    const notes = prompt("Add session notes (optional):");
    
    try {
      const response = await fetch(`/api/appointments/${appointmentId}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          action: 'complete',
          therapistNotes: notes
        }),
      });

      if (response.ok) {
        alert("Appointment marked as completed!");
        fetchAppointments();
      } else {
        const error = await response.json();
        alert(error.error || "Failed to mark appointment as completed");
      }
    } catch (error) {
      console.error("Failed to complete appointment:", error);
      alert("Failed to complete appointment");
    }
  };

  const handleDelete = async (appointmentId) => {
    if (!confirm("Are you sure you want to delete this appointment?")) {
      return;
    }

    try {
      const response = await fetch(`/api/appointments/${appointmentId}`, {
        method: 'DELETE',
      });

      if (response.ok) {
        alert("Appointment deleted successfully!");
        fetchAppointments();
      } else {
        const error = await response.json();
        alert(error.error || "Failed to delete appointment");
      }
    } catch (error) {
      console.error("Failed to delete appointment:", error);
      alert("Failed to delete appointment");
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

  const filteredAppointments = appointments.filter(apt => {
    if (filter === 'all') return true;
    return apt.status === filter;
  });

  const pendingCount = appointments.filter(a => a.status === 'pending').length;
  const acceptedCount = appointments.filter(a => a.status === 'accepted').length;

  if (isLoading) {
    return (
      <div className="flex flex-col items-center justify-center py-12">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600 mb-4"></div>
        <p className="text-gray-600">Loading appointments...</p>
      </div>
    );
  }

  if (error) {
    return (
      <Card>
        <CardContent className="text-center py-8">
          <XCircle className="w-12 h-12 text-red-500 mx-auto mb-4" />
          <p className="text-red-600 font-medium mb-2">Failed to load appointments</p>
          <p className="text-gray-600 text-sm mb-4">{error}</p>
          <Button onClick={fetchAppointments} variant="outline">
            Try Again
          </Button>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {/* Stats Summary */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="pt-6">
            <div className="text-center">
              <p className="text-2xl font-bold text-gray-900">{appointments.length}</p>
              <p className="text-sm text-gray-600">Total Appointments</p>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="text-center">
              <p className="text-2xl font-bold text-yellow-600">{pendingCount}</p>
              <p className="text-sm text-gray-600">Pending Requests</p>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="text-center">
              <p className="text-2xl font-bold text-green-600">{acceptedCount}</p>
              <p className="text-sm text-gray-600">Accepted</p>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="text-center">
              <p className="text-2xl font-bold text-blue-600">
                {appointments.filter(a => a.status === 'completed').length}
              </p>
              <p className="text-sm text-gray-600">Completed</p>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Header with Filter */}
      <Card>
        <CardHeader>
          <CardTitle>Appointment Requests & Sessions</CardTitle>
          <CardDescription>
            Manage your patient appointments and sessions
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex gap-2 flex-wrap">
            <Button
              variant={filter === 'all' ? 'default' : 'outline'}
              onClick={() => setFilter('all')}
              size="sm"
            >
              All ({appointments.length})
            </Button>
            <Button
              variant={filter === 'pending' ? 'default' : 'outline'}
              onClick={() => setFilter('pending')}
              size="sm"
              className="relative"
            >
              Pending ({pendingCount})
              {pendingCount > 0 && (
                <span className="absolute -top-1 -right-1 w-2 h-2 bg-red-500 rounded-full"></span>
              )}
            </Button>
            <Button
              variant={filter === 'accepted' ? 'default' : 'outline'}
              onClick={() => setFilter('accepted')}
              size="sm"
            >
              Accepted ({acceptedCount})
            </Button>
            <Button
              variant={filter === 'declined' ? 'default' : 'outline'}
              onClick={() => setFilter('declined')}
              size="sm"
            >
              Declined
            </Button>
            <Button
              variant={filter === 'completed' ? 'default' : 'outline'}
              onClick={() => setFilter('completed')}
              size="sm"
            >
              Completed
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Appointments List */}
      {filteredAppointments.length === 0 ? (
        <Card>
          <CardContent className="text-center py-8 text-gray-500">
            No appointments found for this filter
          </CardContent>
        </Card>
      ) : (
        <div className="space-y-4">
          {filteredAppointments.map((apt) => (
            <Card key={apt._id} className="overflow-hidden">
              <CardContent className="p-6">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    {/* Patient Info */}
                    <div className="flex items-center gap-2 mb-3">
                      <User className="w-5 h-5 text-gray-500" />
                      <h3 className="font-semibold text-lg">
                        {apt.patientId?.userId?.firstName} {apt.patientId?.userId?.lastName}
                      </h3>
                      <Badge className={getStatusColor(apt.status)}>
                        {apt.status}
                      </Badge>
                    </div>

                    {/* Appointment Details */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-sm text-gray-600 mb-3">
                      <div className="flex items-center gap-2">
                        <Calendar className="w-4 h-4" />
                        <span>
                          {new Date(apt.scheduledAt).toLocaleDateString('en-US', {
                            weekday: 'long',
                            year: 'numeric',
                            month: 'long',
                            day: 'numeric'
                          })}
                        </span>
                      </div>
                      <div className="flex items-center gap-2">
                        <Clock className="w-4 h-4" />
                        <span>
                          {new Date(apt.scheduledAt).toLocaleTimeString('en-US', {
                            hour: '2-digit',
                            minute: '2-digit'
                          })} • {apt.duration} minutes
                        </span>
                      </div>
                      <div className="capitalize">
                        Session Type: {apt.type}
                      </div>
                      <div className="capitalize">
                        Meeting: {apt.meetingType}
                      </div>
                    </div>

                    {/* Patient Notes */}
                    {apt.patientNotes && (
                      <div className="bg-blue-50 p-3 rounded-md mb-3">
                        <p className="text-sm font-medium text-blue-900 mb-1">Patient Notes:</p>
                        <p className="text-sm text-blue-700">{apt.patientNotes}</p>
                      </div>
                    )}

                    {/* Therapist Notes */}
                    {apt.therapistNotes && (
                      <div className="bg-gray-50 p-3 rounded-md mb-3">
                        <p className="text-sm font-medium text-gray-900 mb-1">Your Notes:</p>
                        <p className="text-sm text-gray-700">{apt.therapistNotes}</p>
                      </div>
                    )}

                    {/* Editing Mode */}
                    {editingAppointment === apt._id && (
                      <div className="bg-indigo-50 p-4 rounded-md space-y-3 mb-3">
                        <h4 className="font-medium text-indigo-900">Modify Appointment</h4>
                        
                        {/* Reschedule */}
                        <div className="grid grid-cols-2 gap-3">
                          <div>
                            <label className="text-xs text-gray-600 block mb-1">New Date</label>
                            <Input
                              type="date"
                              value={newDate}
                              onChange={(e) => setNewDate(e.target.value)}
                              min={new Date().toISOString().split('T')[0]}
                            />
                          </div>
                          <div>
                            <label className="text-xs text-gray-600 block mb-1">New Time</label>
                            <Input
                              type="time"
                              value={newTime}
                              onChange={(e) => setNewTime(e.target.value)}
                            />
                          </div>
                        </div>
                        <Button 
                          onClick={() => handleUpdateTime(apt._id)}
                          size="sm"
                          className="w-full"
                        >
                          Update Time
                        </Button>

                        {/* Duration */}
                        <div>
                          <label className="text-xs text-gray-600 block mb-1">Duration (minutes)</label>
                          <select
                            value={newDuration}
                            onChange={(e) => setNewDuration(Number(e.target.value))}
                            className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm"
                          >
                            <option value={30}>30 minutes</option>
                            <option value={45}>45 minutes</option>
                            <option value={60}>60 minutes</option>
                            <option value={90}>90 minutes</option>
                            <option value={120}>120 minutes</option>
                          </select>
                        </div>
                        <Button 
                          onClick={() => handleUpdateDuration(apt._id)}
                          size="sm"
                          className="w-full"
                          variant="secondary"
                        >
                          Update Duration
                        </Button>

                        <Button 
                          onClick={() => setEditingAppointment(null)}
                          size="sm"
                          variant="outline"
                          className="w-full"
                        >
                          Cancel Editing
                        </Button>
                      </div>
                    )}

                    {/* Action Buttons */}
                    <div className="flex flex-wrap gap-2 mt-4">
                      {apt.status === 'pending' && (
                        <>
                          <Button 
                            onClick={() => handleAccept(apt._id)}
                            size="sm"
                            className="bg-green-600 hover:bg-green-700"
                          >
                            <CheckCircle className="w-4 h-4 mr-1" />
                            Accept
                          </Button>
                          <Button 
                            onClick={() => handleDecline(apt._id)}
                            size="sm"
                            variant="destructive"
                          >
                            <XCircle className="w-4 h-4 mr-1" />
                            Decline
                          </Button>
                        </>
                      )}

                      {(apt.status === 'accepted' || apt.status === 'pending') && (
                        <Button 
                          onClick={() => {
                            setEditingAppointment(apt._id);
                            setNewDuration(apt.duration);
                            const date = new Date(apt.scheduledAt);
                            setNewDate(date.toISOString().split('T')[0]);
                            setNewTime(date.toTimeString().slice(0, 5));
                          }}
                          size="sm"
                          variant="outline"
                        >
                          <Edit2 className="w-4 h-4 mr-1" />
                          Modify
                        </Button>
                      )}

                      {apt.status === 'accepted' && new Date(apt.scheduledAt) < new Date() && (
                        <Button 
                          onClick={() => handleComplete(apt._id)}
                          size="sm"
                          className="bg-blue-600 hover:bg-blue-700"
                        >
                          Mark Complete
                        </Button>
                      )}

                      <Button 
                        onClick={() => handleDelete(apt._id)}
                        size="sm"
                        variant="ghost"
                        className="text-red-600 hover:text-red-700 hover:bg-red-50"
                      >
                        <Trash2 className="w-4 h-4 mr-1" />
                        Delete
                      </Button>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}
